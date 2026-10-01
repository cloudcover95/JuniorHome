/** Ticket reader. No network. Metro is not started by this file. */
export const TICKET = "ticket.json";
export function readTicket(raw: string) {
  const row = JSON.parse(raw);
  if (row.bind !== "127.0.0.1" || row.model_pull !== false) {
    return { ok: false };
  }
  return { ok: true, kind: row.kind, sha3: row.sha3, pack5: row.pack5 };
}
