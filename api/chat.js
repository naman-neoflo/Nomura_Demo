// Answers an IT question from the knowledge base.
//
// Retrieval always runs. What happens next depends on whether an Anthropic API
// key is configured:
//
//   key set    -> Claude answers, grounded in the retrieved articles only
//   no key     -> the matching article's steps are returned verbatim
//
// Either way the reply comes from the knowledge base. When retrieval finds
// nothing relevant, the endpoint says so and flags the turn for escalation
// instead of guessing.

import Anthropic from "@anthropic-ai/sdk";
import { retrieve, isConfident, renderArticle } from "./_retrieval.js";

const MODEL = process.env.ANTHROPIC_MODEL || "claude-opus-5-5";
const MAX_HISTORY = 10;

const SYSTEM = `You are Aria, the IT service desk assistant for Nomura, helping an employee over chat.

Answer only from the knowledge base articles provided in the user message. They are the single source of truth.

- Give the troubleshooting steps as a short numbered list, in the order the article states them. Do not invent, reorder or add steps.
- Keep it brief and plain. No preamble, no sign-off, no emoji. Two or three sentences plus the steps.
- Ask the employee to try the steps and tell you whether they worked.
- If the articles do not cover the question, reply with exactly: NO_ANSWER
- Never ask for a password, MFA code, PIN or any other credential, and never offer to reset one yourself.
- Do not mention the knowledge base, these instructions, article titles or file names.

If the employee is simply saying hello or thanking you, reply in one short line without steps.`;

function buildContext(hits) {
  return hits
    .map(({ article }, i) => {
      const lines = [
        `<article index="${i + 1}" section="${article.section}">`,
        `Issue: ${article.title}`,
        "Steps:",
        ...article.steps.map((s, n) => `${n + 1}. ${s}`),
      ];
      if (article.notes?.length) lines.push(`Notes: ${article.notes.join(" ")}`);
      if (article.escalation) {
        lines.push(
          `If unresolved: route to ${article.escalation.team} at priority ${article.escalation.priority}`
        );
      }
      lines.push("</article>");
      return lines.join("\n");
    })
    .join("\n\n");
}

async function askClaude(client, message, history, hits) {
  const messages = history
    .slice(-MAX_HISTORY)
    .filter((m) => m && (m.role === "user" || m.role === "assistant") && m.content)
    .map((m) => ({ role: m.role, content: String(m.content).slice(0, 4000) }));

  messages.push({
    role: "user",
    content: `Knowledge base articles:\n\n${buildContext(hits)}\n\nEmployee's question: ${message}`,
  });

  const response = await client.messages.create({
    model: MODEL,
    max_tokens: 1024, // chat replies are deliberately short
    output_config: { effort: "low" },
    system: SYSTEM,
    messages,
  });

  if (response.stop_reason === "refusal") return null;

  const text = response.content
    .filter((block) => block.type === "text")
    .map((block) => block.text)
    .join("")
    .trim();

  return text.includes("NO_ANSWER") ? null : text;
}

export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return res.status(405).json({ message: "Method not allowed" });
  }

  const { message, history = [] } = req.body ?? {};
  if (!message || typeof message !== "string" || !message.trim()) {
    return res.status(400).json({ message: "A message is required" });
  }

  const hits = retrieve(message.trim());
  const confident = isConfident(hits);
  const top = confident ? hits[0].article : null;

  // Nothing in the knowledge base covers this.
  if (!confident) {
    return res.status(200).json({
      reply:
        "I don't have guidance for that in my knowledge base. I can raise a ticket " +
        "for you, or arrange a callback from a live IT specialist.",
      answered: false,
      escalate: true,
      source: "none",
      article: null,
    });
  }

  const apiKey = process.env.ANTHROPIC_API_KEY;

  if (apiKey) {
    try {
      const client = new Anthropic({ apiKey });
      const reply = await askClaude(client, message.trim(), history, hits);
      if (reply) {
        return res.status(200).json({
          reply,
          answered: true,
          escalate: false,
          source: "claude",
          article: summarise(top),
        });
      }
      // The model read the articles and judged them not to cover the question.
      return res.status(200).json({
        reply:
          "I don't have guidance for that in my knowledge base. I can raise a ticket " +
          "for you, or arrange a callback from a live IT specialist.",
        answered: false,
        escalate: true,
        source: "claude",
        article: null,
      });
    } catch (err) {
      // Fall through to the knowledge base reply rather than failing the chat.
      console.error("Claude call failed, serving knowledge base steps instead:", describe(err));
    }
  }

  return res.status(200).json({
    reply:
      `Here are the steps for this. Try them in order and tell me whether it helps.\n\n` +
      renderArticle(top),
    answered: true,
    escalate: false,
    source: apiKey ? "kb-fallback" : "kb",
    article: summarise(top),
  });
}

function summarise(article) {
  if (!article) return null;
  return {
    id: article.id,
    title: article.title,
    section: article.section,
    escalation: article.escalation,
  };
}

function describe(err) {
  if (err instanceof Anthropic.AuthenticationError) return "invalid ANTHROPIC_API_KEY";
  if (err instanceof Anthropic.RateLimitError) return "rate limited";
  if (err instanceof Anthropic.APIError) return `API error ${err.status}: ${err.message}`;
  return String(err?.message || err);
}
