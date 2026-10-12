# Nomura IT Service Desk — Callback Request

A single-page callback request form for the IT service desk, styled to Nomura's
visual identity. Submitting the form POSTs the caller's details to an n8n
webhook, which places an outbound AI voice call via Vapi.

## Files

| File | Purpose |
| --- | --- |
| `callback.html` | The form. Self-contained — no build step, no dependencies. |
| `server.py` | Local dev server. Serves the page and proxies submissions to n8n. |

## Running locally

```bash
python3 server.py        # http://localhost:8000
```

The page posts to `/api/callback`, and `server.py` forwards that to the webhook
**server-side**. This exists because the browser cannot call the n8n webhook
directly: the preflight `OPTIONS` request returns `500` with no
`Access-Control-Allow-Origin` header, so the browser blocks the POST and
reports only a generic "Failed to fetch".

Opening `callback.html` straight from disk makes it post to n8n directly, which
will keep failing on CORS until the webhook sends the right headers.

## Submitted payload

```json
{
  "name": "Naman Sharma",
  "contact_number": "+12125550147",
  "reason": "Password Reset",
  "submitted_at": "2026-10-13T02:16:00.000Z"
}
```

`reason` is one of: Password Reset, Software Installation, General IT Query,
Account Unlock, VPN Not Connecting.

`contact_number` is normalised to E.164 before sending: the page strips spaces,
brackets and dashes, so `+1 (212) 555-0147` is submitted as `+12125550147`.
Any country code is accepted — India, US and others — but it must be present.

## n8n configuration

Settings that have to be right on the Webhook node, each of which produced a
real failure during setup:

- **HTTP Method: POST.** While set to GET, a POST returns
  `404 "This webhook is not registered for POST requests."`
- **Respond: Immediately**, or add a *Respond to Webhook* node. Otherwise:
  `{"code":0,"message":"No Respond to Webhook node found in the workflow"}`
- **Allowed Origins (CORS):** set this to the page's domain when the page is
  hosted somewhere real. Then `server.py` is no longer needed.

In the downstream HTTP Request node that calls Vapi, wrap every expression so
it emits valid JSON — free-text input containing a quote would otherwise break
the body:

```json
{
  "assistantId": "…",
  "assistantOverrides": {
    "variableValues": {
      "name": {{ JSON.stringify($json.body.name) }},
      "phone": {{ JSON.stringify($json.body.contact_number) }},
      "reason": {{ JSON.stringify($json.body.reason) }}
    }
  },
  "phoneNumberId": "…",
  "customer": { "number": {{ JSON.stringify($json.body.contact_number) }} }
}
```

The JSON Body field must be in **Expression** mode, or n8n sends the `{{ }}`
through as literal text.

## Notes

- A successful submission **places a real phone call** to the number entered.
  Test with a number you control.
- The sidebar content (service desk hours, callback time estimate) is
  placeholder text — replace it with real values before use.
- Internal tool. It carries Nomura's branding without being an official Nomura
  property, so keep this repo and any deployment private.
