import { defineFunction, secret } from "@aws-amplify/backend";

/**
 * Holds the third-party news API key so it never enters the agent's deployment package, its
 * prompt, or its context. That boundary is the whole reason this is a separate function rather
 * than an HTTP call from the agent.
 *
 * Set the key with:  npx ampx sandbox secret set NEWS_API_KEY
 * Amplify resolves it into this function's environment at deploy time — narrow custody, not
 * encryption at rest, so keep the function's role minimal and never log the URL.
 */
export const getNews = defineFunction({
  name: "getNews",
  entry: "./handler.ts",
  timeoutSeconds: 10, // one third-party fetch inside an agent turn — fail fast, don't hang
  environment: {
    NEWS_API_KEY: secret("NEWS_API_KEY"),
  },
});
