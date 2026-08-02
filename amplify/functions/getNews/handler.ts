import { env } from "$amplify/env/getNews";

/**
 * Recent headlines, fetched on the agent's behalf.
 *
 * Hardened against four things a security review found in the original version of this
 * function, all of which matter more once an AGENT is choosing the arguments:
 *
 *  1. The topic is an allow-listed key, not a free search string. An injected instruction must
 *     not be able to drive arbitrary metered queries, or push the user's own words to a third
 *     party.
 *  2. The constructed URL — which contains the API key — is never logged and never thrown. An
 *     error whose `cause` carries the URL puts the key in CloudWatch.
 *  3. `articles[0]` is not assumed. On a 401/429, or `{"status":"error"}`, the original threw
 *     a TypeError; now it returns an empty list.
 *  4. The timeout is 10s, not 500s. A synchronous third-party call sits inside an agent turn.
 */

const TOPICS: Record<string, string> = {
  diabetes: "diabetes",
  insulin: "insulin",
  public_health: "CDC public health",
};

const MAX_ITEMS = 5;
const TIMEOUT_MS = 8000;

type Article = { title?: string; description?: string; publishedAt?: string };

export const handler = async (event: {
  arguments?: { category?: string };
}): Promise<{ items: Article[]; error?: string }> => {
  const requested = event?.arguments?.category ?? "";
  const query = TOPICS[requested];

  if (!query) {
    // Do not echo the requested value: it may be attacker-influenced text, and reflecting it
    // launders it back through a channel the model trusts.
    return { items: [], error: `Unknown topic. Choose one of: ${Object.keys(TOPICS).join(", ")}` };
  }

  // `searchIn=title` matters: without it the term matches anywhere in the body, and a query for
  // "insulin" returns loosely-related health news that mentions the word once. A headline the
  // agent labels LIVE should at least be about the topic it was asked for.
  const url =
    `https://newsapi.org/v2/everything?q=${encodeURIComponent(query)}` +
    `&searchIn=title&language=en&sortBy=publishedAt&pageSize=${MAX_ITEMS}` +
    `&apiKey=${env.NEWS_API_KEY}`;

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);

  try {
    const response = await fetch(url, { signal: controller.signal });
    if (!response.ok) {
      // Status only — never the URL, never the body, either of which can carry the key.
      console.error("news upstream returned", response.status);
      return { items: [], error: "The news source is unavailable." };
    }

    const body = (await response.json()) as { articles?: Article[] };
    const articles = Array.isArray(body.articles) ? body.articles : [];

    return {
      items: articles.slice(0, MAX_ITEMS).map((a) => ({
        title: (a.title ?? "").slice(0, 200),
        description: (a.description ?? "").slice(0, 400),
        publishedAt: a.publishedAt ?? "",
      })),
    };
  } catch (error) {
    // Log the ERROR NAME only. The message or `cause` of a fetch failure can include the URL.
    console.error("news fetch failed", error instanceof Error ? error.name : "unknown");
    return { items: [], error: "The news source is unavailable." };
  } finally {
    clearTimeout(timer);
  }
};
