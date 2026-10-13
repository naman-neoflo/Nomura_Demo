// Raises a support ticket from the chat widget.
//
// Routing (team + priority) comes from the knowledge base article the chat was
// looking at, so a ticket inherits the escalation the KB specifies. When the
// article is unknown the KB's own default applies: IT Service Desk, P3.
//
// A ticket is only real if TICKET_WEBHOOK_URL points at something that records
// it. With no webhook configured this endpoint returns 503 rather than inventing
// a reference number, so the widget can offer the callback form instead of
// telling someone their ticket was filed when it was not.

const TICKET_WEBHOOK = process.env.TICKET_WEBHOOK_URL;

const DEFAULT_ROUTE = { team: "IT Service Desk", priority: "P3" };

export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return res.status(405).json({ message: "Method not allowed" });
  }

  const { name, contact, issue, article, transcript = [] } = req.body ?? {};

  if (!issue || !String(issue).trim()) {
    return res.status(400).json({ message: "A description of the issue is required" });
  }

  if (!TICKET_WEBHOOK) {
    return res.status(503).json({
      message: "Ticketing is not connected yet",
      configured: false,
    });
  }

  const route = article?.escalation
    ? { team: article.escalation.team, priority: article.escalation.priority }
    : DEFAULT_ROUTE;

  const payload = {
    source: "it-service-desk-chat",
    name: name || null,
    contact_number: contact || null,
    issue: String(issue).trim(),
    kb_article: article?.title || null,
    kb_article_id: article?.id || null,
    assignment_group: route.team,
    priority: route.priority,
    transcript: transcript.slice(-20),
    created_at: new Date().toISOString(),
  };

  try {
    const upstream = await fetch(TICKET_WEBHOOK, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const text = await upstream.text();
    if (!upstream.ok) {
      console.error("Ticket webhook returned", upstream.status, text.slice(0, 300));
      return res.status(502).json({ message: "The ticketing system rejected the request" });
    }

    // Use the reference the ticketing system returns; never fabricate one.
    let reference = null;
    try {
      const data = JSON.parse(text);
      reference = data.reference || data.ticket || data.number || data.id || null;
    } catch {
      /* upstream returned something other than JSON */
    }

    return res.status(200).json({
      configured: true,
      reference,
      team: route.team,
      priority: route.priority,
    });
  } catch (err) {
    console.error("Ticket webhook unreachable:", err?.message || err);
    return res.status(502).json({ message: "Could not reach the ticketing system" });
  }
}
