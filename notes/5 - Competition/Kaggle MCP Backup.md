Kaggle MCP server config - removed from ~/.claude.json on 2026-04-24
To restore, paste into "mcpServers" block under "c:/Users/nitro/Desktop/ARC" in ~/.claude.json

"kaggle": {
  "type": "stdio",
  "command": "npx",
  "args": [
    "mcp-remote",
    "https://www.kaggle.com/mcp",
    "--header",
    "Authorization: Bearer KGAT_d03b59934c3b904006f2c3b576e562e1"
  ],
  "env": {}
}
