# Google Custom Search API Setup Troubleshooting

The error "This project does not have the access to Custom Search JSON API" means:

## What to check:

1. **Verify the Custom Search API is enabled:**
   - Go to https://console.cloud.google.com
   - Make sure you're in the CORRECT PROJECT (check top left)
   - Go to "APIs & Services" > "Enabled APIs & services"
   - Search for "Custom Search"
   - If you see "Custom Search API" with a BLUE checkmark, it's enabled
   - If it's not in the list OR shows "ENABLE" button, you need to enable it

2. **Check your API Key credentials:**
   - Go to "APIs & Services" > "Credentials"
   - You should see an API Key listed
   - Click on it and verify under "API restrictions" that "Custom Search API" is selected

3. **Verify your Search Engine (CX):**
   - Go to https://programmablesearchengine.google.com/
   - Log in with the SAME Google account
   - Click "My search engines"
   - Find your search engine and copy the "Search engine ID"
   - Make sure it's the same as in your .env file

4. **Important: The API key's associated project must be the SAME project where you enabled Custom Search API**

## Common issues:

- Using an API key from a different project
- Custom Search API is created but not ENABLED
- Using a personal Google account key instead of a service account
- Free quota might be exhausted (100/day)

## Try this:

1. Create a NEW API Key fresh just for Custom Search
2. Make sure it's in the SAME project where you enabled Custom Search API
3. Update your .env file
4. Restart the app
