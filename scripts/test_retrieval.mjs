// Checks retrieval against how people actually phrase helpdesk questions.
//   node scripts/test_retrieval.mjs
import { retrieve, RELEVANCE_FLOOR } from "../api/_retrieval.js";

// [question, expected article title or null when it should not be answered]
const CASES = [
  ["my outlook won't open", "Outlook Not Opening"],
  ["outlook keeps asking for my password", "Outlook Password Keeps Asking"],
  ["emails are stuck in my outbox", "Emails Stuck in Outbox"],
  ["I'm not receiving any emails", "Outlook Not Syncing / Emails Not Receiving"],
  ["important mail is going to junk", "Emails Going to Junk / Spam"],
  ["teams is stuck on the loading screen", "Teams Not Loading / Stuck on Loading Screen"],
  ["my teams calls keep dropping", "Teams Calls Dropping or Poor Audio"],
  ["I can't join a teams meeting", "Cannot Join a Teams Meeting"],
  ["vpn won't connect", "VPN Not Connecting"],
  ["vpn keeps disconnecting every few minutes", "VPN Disconnects Frequently"],
  ["connected to vpn but can't reach the share", "VPN Connects but Cannot Access Internal Resources"],
  ["no internet at my desk", "No Internet Access in Office"],
  ["the office internet is really slow", "Slow Internet in Office"],
  ["I can't see the nomura wifi network", "WiFi Not Showing Nomura Network"],
  ["my laptop is really slow", "Laptop Running Slow"],
  ["laptop screen keeps flickering", "Laptop Screen Flickering"],
  ["my laptop isn't charging", "Laptop Not Charging"],
  ["second monitor not detected", "External Monitor Not Detected"],
  ["my mouse stopped working", "Keyboard or Mouse Not Working"],
  ["can't get to the shared drive", "Cannot Access a Shared Drive or Folder"],
  ["mfa isn't working", "Multi-Factor Authentication (MFA) Not Working"],
  ["printer says offline", "Cannot Print — Printer Offline"],
  ["print jobs are stuck", "Print Jobs Stuck in Queue"],
  ["what is a p1 ticket", "Ticket priority levels and response times"],
  // Out of scope - should fall below the floor.
  ["how do I book annual leave", null],
  ["what is the share price of nomura", null],
  ["can you order me a new desk chair", null],
];

let top1 = 0, top3 = 0, scoped = 0, inScope = 0, outOk = 0, outTotal = 0;

for (const [question, expected] of CASES) {
  const hits = retrieve(question);
  const top = hits[0];
  const confident = top && top.score >= RELEVANCE_FLOOR;
  const answered = confident ? top.article.title : null;

  if (expected === null) {
    outTotal++;
    const ok = answered === null;
    if (ok) outOk++;
    console.log(`${ok ? "ok  " : "FAIL"} ${(top ? top.score.toFixed(2) : "-").padStart(6)}  [out of scope] ${question}`);
    if (!ok) console.log(`       answered with: ${answered}`);
    continue;
  }

  inScope++;
  const hit1 = answered === expected;
  const hit3 = confident && hits.some((h) => h.article.title === expected);
  if (hit1) top1++;
  if (hit3) top3++;
  const mark = hit1 ? "ok  " : hit3 ? "top3" : "FAIL";
  console.log(`${mark} ${top.score.toFixed(2).padStart(6)}  ${question}`);
  if (!hit1) console.log(`       expected: ${expected}\n       got:      ${answered}`);
}

console.log(`
top-1 (used when no LLM key is set): ${top1}/${inScope}
top-3 (what the model receives):     ${top3}/${inScope}
out-of-scope correctly declined:     ${outOk}/${outTotal}
relevance floor: ${RELEVANCE_FLOOR}`);

process.exit(top3 === inScope && outOk === outTotal ? 0 : 1);
