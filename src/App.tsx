import { useState } from "react";
import {
  Authenticator,
  Badge,
  Button,
  Card,
  Divider,
  Flex,
  Heading,
  Loader,
  Text,
  TextAreaField,
  View,
} from "@aws-amplify/ui-react";
import { generateClient } from "aws-amplify/data";
import type { Schema } from "../amplify/data/resource";

const client = generateClient<Schema>();

/**
 * The badge is the point of the whole demo. An answer's trustworthiness depends on which tool
 * produced it, and the user cannot see the tools — so the mode is shown, not implied.
 */
const MODES: Record<string, { label: string; colour: "success" | "info" | "warning" | "error"; note: string }> = {
  agent: { label: "ANSWERED", colour: "success", note: "figures verified or computed" },
  "withheld-ungrounded": {
    label: "WITHHELD",
    colour: "error",
    note: "contained a figure with no verified source",
  },
  fallback: { label: "FALLBACK", colour: "warning", note: "model unavailable — answered from the bundle" },
  rejected: { label: "REJECTED", colour: "warning", note: "input rejected before any model call" },
  unconfigured: { label: "UNCONFIGURED", colour: "warning", note: "the agent runtime is not wired up" },
  error: { label: "ERROR", colour: "error", note: "the agent is unavailable" },
};

const SAMPLES = [
  { label: "Verified figure", q: "What percent of U.S. adults with diagnosed diabetes currently take insulin?" },
  { label: "Computed breakdown", q: "Break down insulin use among diagnosed diabetics by sex" },
  { label: "Methodology (retrieved)", q: "How does survey weighting work in the NHIS?" },
  { label: "Live news", q: "What is new in diabetes news?" },
  { label: "Off-topic → refuse", q: "What percent of U.S. adults have asthma?" },
  { label: "Medical advice → refuse", q: "I have diabetes, should I start taking insulin?" },
];

type Result = Awaited<ReturnType<typeof client.queries.askAgent>>["data"];

/** Render the model's **bold** without pulling in a markdown library — the figures are the
 *  point of every answer, and showing them as literal asterisks buries them. */
function RichText({ text }: { text: string }) {
  return (
    <>
      {text.split(/(\*\*[^*]+\*\*)/g).map((part, i) =>
        part.startsWith("**") && part.endsWith("**") ? (
          <strong key={i}>{part.slice(2, -2)}</strong>
        ) : (
          <span key={i}>{part}</span>
        )
      )}
    </>
  );
}

export default function App() {
  return <Authenticator>{({ signOut }) => <Ask signOut={signOut} />}</Authenticator>;
}

function Ask({ signOut }: { signOut?: () => void }) {
  const [question, setQuestion] = useState(SAMPLES[0].q);
  const [result, setResult] = useState<Result>(null);
  const [elapsed, setElapsed] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function ask() {
    setLoading(true);
    setResult(null);
    setError(null);
    setElapsed(null);
    const started = performance.now();
    try {
      const { data, errors } = await client.queries.askAgent({ question });
      setElapsed(Math.round(performance.now() - started));
      if (errors?.length) setError(errors.map((e) => e.message).join("; "));
      else setResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  const mode = result?.mode ? MODES[result.mode] ?? MODES.agent : null;

  return (
    <View maxWidth="760px" margin="0 auto" padding="1.5rem">
      <Flex justifyContent="space-between" alignItems="center">
        <Heading level={3}>Four Kinds of Knowing · NHIS 2023</Heading>
        <Button size="small" variation="link" onClick={signOut}>
          Sign out
        </Button>
      </Flex>
      <Text color="font.tertiary" fontSize="0.9rem">
        Four tools — verified figures, computed queries, retrieved documentation, live news — and
        a rule enforced in code: a number may only come from a verified or computed source.
      </Text>
      <Divider marginBlock="1rem" />

      <Flex wrap="wrap" gap="0.5rem" marginBottom="1rem">
        {SAMPLES.map((s) => (
          <Button
            key={s.q}
            size="small"
            variation={question === s.q ? "primary" : undefined}
            onClick={() => setQuestion(s.q)}
          >
            {s.label}
          </Button>
        ))}
      </Flex>

      <TextAreaField
        label="Question"
        value={question}
        rows={2}
        onChange={(e) => setQuestion(e.target.value)}
      />

      <Flex alignItems="center" gap="0.75rem" marginTop="0.75rem">
        <Button variation="primary" onClick={ask} isDisabled={loading}>
          Ask
        </Button>
        {loading && <Loader />}
      </Flex>

      {error && (
        <Text color="font.error" marginTop="1rem">
          {error}
        </Text>
      )}

      {result && mode && (
        <Card variation="outlined" marginTop="1.25rem">
          <Flex alignItems="center" gap="0.5rem" marginBottom="0.25rem" wrap="wrap">
            <Badge variation={mode.colour}>{mode.label}</Badge>
            {elapsed !== null && <Badge>{elapsed} ms round-trip</Badge>}
          </Flex>
          <Text fontSize="0.78rem" color="font.tertiary" marginBottom="0.75rem">
            {mode.note}
          </Text>
          <Text whiteSpace="pre-wrap">
            <RichText text={result.answer ?? ""} />
          </Text>
        </Card>
      )}
    </View>
  );
}
