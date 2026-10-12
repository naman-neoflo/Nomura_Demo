// Vercel serverless function: forwards a callback request to the n8n webhook.
//
// The browser cannot call the webhook directly — its preflight OPTIONS returns
// 500 with no Access-Control-Allow-Origin, so the POST is blocked before it is
// sent. Forwarding server-side sidesteps CORS entirely, and keeps the webhook
// URL out of the client bundle.
//
// Mirrors what server.py does for local development.

const WEBHOOK = process.env.WEBHOOK_URL
  || "https://n8n-sprint.neoflo.ai/webhook/initiate";

export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return res.status(405).json({ message: "Method not allowed" });
  }

  try {
    const upstream = await fetch(WEBHOOK, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req.body ?? {}),
    });

    const text = await upstream.text();
    res.status(upstream.status);
    res.setHeader(
      "Content-Type",
      upstream.headers.get("content-type") || "application/json"
    );
    return res.send(text);
  } catch (err) {
    console.error("Forward to n8n failed:", err);
    return res.status(502).json({ message: "Could not reach the callback service" });
  }
}
