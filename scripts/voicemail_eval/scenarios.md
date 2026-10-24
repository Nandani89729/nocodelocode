# Voicemail detection — call scenarios

Twenty scripted calls. You place an outbound call from a Dograh workflow with
`voicemail_detection.enabled = true`, answer it yourself, and read the **opening
line** out loud. Because you know what you were pretending to be, ground truth
is exact — no labelling pass is needed afterwards.

`harvest.py` parses the table below for the `scenario_id → label` mapping, so the
first two columns are load-bearing: keep the `xx-NN` id format and the label
spelled exactly `human`, `voicemail` or `carrier`.

## How to run a scenario

1. Trigger the outbound call to your own number.
2. Answer, wait for the pause, then read the opening line in the manner described.
3. Stay on until the agent either speaks normally or hangs up.
4. Note the `workflow_run_id` against the scenario id in `runs.csv`.

**Read the line as written.** The point is to test the classifier against a fixed
stimulus, not to improvise — if you paraphrase, the dataset stops being comparable
across reruns and across model variants.

## Why these twenty

Four easy of each class establish that nothing is fundamentally broken. The value
is in the twelve after that. A classifier that only ever sees "Hi, you've reached
Sarah, leave a message" is being asked the question both gpt-4.1 and Jev already
answer correctly; it tells you nothing about which to ship. The hard and ambiguous
rows are where a difference can actually show up, and the ambiguous block is where
Jev's confidence gate is supposed to earn its place by *declining* to answer and
waiting for more speech, which the current forced-binary classifier cannot do.

## Scenarios

| id | label | opening line | how to deliver it |
| --- | --- | --- | --- |
| vm-01 | voicemail | "Hi, you've reached Sarah. I can't take your call right now, please leave a message after the tone." | Flat, recorded cadence. The canonical case. |
| vm-02 | voicemail | "You have reached the voicemail of the accounts team at Brightline. Please leave your name and number." | Business greeting, unhurried. |
| vm-03 | voicemail | "I'm not available right now. Leave a message and I'll get back to you as soon as I can." | No name given — tests reliance on the name pattern. |
| vm-04 | voicemail | "Hey, it's Dan. You know what to do." | Very short, informal voicemail. No explicit "leave a message". |
| hu-01 | human | "Hello?" | Rising, questioning. The single hardest easy case. |
| hu-02 | human | "Yeah?" | Curt. |
| hu-03 | human | "Priya speaking." | Brisk and professional — resembles a business greeting. |
| hu-04 | human | "Hello, who's this?" | Slightly suspicious, as to an unknown number. |
| hd-01 | human | "Hello, you've reached me, go ahead." | **Lexical trap.** A real person using voicemail phrasing. |
| hd-02 | voicemail | "Hi." *(pause 2s)* "Sorry I missed you, leave a message." | The first second is indistinguishable from a human. Tests whether a verdict is committed too early. |
| hd-03 | human | "Hello? Sorry, one second, it's really loud here." | Speak with background noise — TV, traffic, music. Tests ASR degradation. |
| hd-04 | human | *(silence ~3s)* "…hello?" | Say nothing at all first. Tests behaviour when the opening is empty. |
| hd-05 | carrier | "The number you have dialled is not in service. Please check the number and try again." | Flat, synthetic delivery. |
| hd-06 | carrier | "The mailbox belonging to this number is full and cannot accept new messages." | Flat, synthetic delivery. |
| hd-07 | voicemail | "HiyouvereachedRaj,I'mnotaroundrightnow,leaveamessage." | Deliberately fast and accented — stresses the transcript quality the classifier depends on. |
| hd-08 | human | "Hi, can you call me back in ten minutes? I'm driving." | **Trap.** Callback phrasing from a live person. |
| am-01 | human | "Hi." | One word, nothing else. Say nothing further for 5s. Genuinely undecidable at first. |
| am-02 | voicemail | "Hello." *(pause 3s)* "…Leave a message." | Mirror of am-01 with the opposite label and an identical opening. |
| am-03 | human | "Hello, hello? Can you hear me?" | Connection-trouble phrasing, which reads oddly in text. |
| am-04 | voicemail | "Thanks for calling. Nobody is available to take your call." | No "message", no name, no tone reference. |

## Notes

- `am-01` and `am-02` are a matched pair with near-identical openings and
  opposite labels. Neither classifier can be right about both from the first
  second. The interesting question is whether Jev *holds* on them (low
  confidence, no verdict emitted, waits for more transcript) while gpt-4.1 is
  forced to guess. If Jev guesses here too, the confidence gate isn't working.
- `hd-05` and `hd-06` are labelled `carrier`. The live pipeline collapses that
  to `VOICEMAIL`, so they count as correct for either classifier when treated as
  voicemail — but Jev reports them separately, which is worth recording for
  future retry logic.
- If a scenario produces no captured user speech at all, `harvest.py` flags it.
  Re-run that one; an empty transcript tests nothing.
