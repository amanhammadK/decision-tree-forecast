import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { processItem } from "./core.js";
const server = new Server({ name: "decision-tree-forecast", version: "1.0.0" }, { capabilities: { tools: {} } });
server.setRequestHandler("tools/list", async () => ({
  tools: [{
    name: "train_and_predict",
    description: "Train a decision tree regressor and make predictions",
    inputSchema: {
      type: "object",
      properties: {
        input: {
          type: "string",
          description: "JSON string with features (2D array), targets (1D array), and optional predict_features"
        },
        params: {
          type: "object",
          properties: {
            maxDepth: { type: "number", description: "Maximum tree depth" },
            minSamplesSplit: { type: "number", description: "Min samples to split a node" },
            minSamplesLeaf: { type: "number", description: "Min samples in a leaf" }
          }
        }
      },
      required: ["input"]
    }
  }]
}));
server.setRequestHandler("tools/call", async (req) => {
  const { name, arguments: args } = req.params;
  if (name === "train_and_predict") return { content: [{ type: "text", text: JSON.stringify(await processItem(args.input, args.params)) }] };
  throw new Error("Unknown tool: " + name);
});
export { server };
