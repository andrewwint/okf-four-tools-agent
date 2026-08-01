import { defineAuth } from "@aws-amplify/backend";

/**
 * Email sign-in. Note what this does NOT give you: "authenticated" is not "authorised to
 * spend". Self-signup is open, so for anything beyond a demo pair this with a budget alarm
 * (or an admin-create-only pool) — a four-tool agent loop is several model calls per question.
 */
export const auth = defineAuth({
  loginWith: { email: true },
});
