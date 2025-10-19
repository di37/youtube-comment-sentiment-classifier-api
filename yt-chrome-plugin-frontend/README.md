# YouTube Sentiment Insights - Chrome Extension

A Chrome extension that analyzes YouTube video comments for sentiment using AI-powered sentiment analysis.

## 🌟 Features

- **Real-time Sentiment Analysis**: Analyzes YouTube comments and categorizes them as Positive, Neutral, or Negative
- **Visual Analytics**:
  - Pie chart showing sentiment distribution
  - Word cloud from all comments
  - Sentiment trend graph over time
- **Comprehensive Metrics**:
  - Total comments analyzed
  - Unique commenters
  - Average comment length
  - Average sentiment score
- **Top Comments Display**: Shows the top 25 comments with their sentiment labels

## 📋 Prerequisites

1. **API Server Running**: Your sentiment analysis API must be running at `http://40.172.234.207:6091`
2. **YouTube Data API Key**: The extension uses YouTube Data API v3 to fetch comments

## 🚀 Installation

### Step 1: Load Extension in Chrome

1. **Open Chrome** and navigate to `chrome://extensions/`

2. **Enable Developer Mode**:
   - Toggle the "Developer mode" switch in the top right corner

3. **Load Unpacked Extension**:
   - Click "Load unpacked" button
   - Navigate to the `yt-chrome-plugin-frontend` directory
   - Select the folder and click "Open"

4. **Verify Installation**:
   - You should see "YouTube Sentiment Insights" in your extensions list
   - The extension icon should appear in your Chrome toolbar

### Step 2: Pin the Extension (Optional)

1. Click the puzzle icon (Extensions) in Chrome toolbar
2. Find "YouTube Sentiment Insights"
3. Click the pin icon to keep it visible

## 🎯 How to Use

1. **Navigate to YouTube**:
   - Go to https://www.youtube.com
   - Open any video with comments

2. **Activate the Extension**:
   - Click the "YouTube Sentiment Insights" icon in your toolbar
   - The extension will automatically detect the video ID

3. **Wait for Analysis**:
   - The extension will:
     - Fetch up to 500 comments
     - Send them to the AI API for sentiment analysis
     - Generate visualizations
     - Display results

4. **View Results**:
   - Comment Analysis Summary (metrics)
   - Sentiment Distribution (pie chart)
   - Sentiment Trend Over Time (line graph)
   - Comment Word Cloud
   - Top 25 Comments with sentiments

## 🔧 Configuration

### Changing the API URL

If your API is hosted at a different URL, update `popup.js`:

```javascript
const API_URL = 'http://YOUR_SERVER_IP:6091';
```

### Changing YouTube API Key

If you have your own YouTube Data API key, update `popup.js`:

```javascript
const API_KEY = 'YOUR_YOUTUBE_API_KEY';
```

To get your own YouTube Data API key:
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable "YouTube Data API v3"
4. Create credentials (API Key)
5. Copy the API key and paste it in the extension

## 📊 API Endpoints Used

The extension communicates with these API endpoints:

- `POST /predict_with_timestamps` - Batch sentiment prediction with timestamps
- `POST /generate_chart` - Generate sentiment distribution pie chart
- `POST /generate_wordcloud` - Generate word cloud from comments
- `POST /generate_trend_graph` - Generate sentiment trend over time

## 🛡️ Permissions

The extension requires the following permissions:

- `tabs` - To access the current tab's URL
- `activeTab` - To interact with the active YouTube page
- `scripting` - For potential future features
- `host_permissions`:
  - `http://localhost/*` - For local development
  - `http://40.172.234.207/*` - For accessing the deployed API
  - `https://www.googleapis.com/*` - For YouTube Data API

## 🐛 Troubleshooting

### Extension doesn't show results

1. **Check API Server**:
   ```bash
   curl http://40.172.234.207:6091/health
   ```
   Should return: `{"status":"healthy"}`

2. **Check Browser Console**:
   - Right-click extension popup → "Inspect"
   - Check Console tab for errors

3. **Verify YouTube URL**:
   - Make sure you're on a valid YouTube video page
   - URL should match: `https://www.youtube.com/watch?v=VIDEO_ID`

### CORS Errors

If you see CORS errors in the console:
- Ensure the API has CORS enabled (it should be already)
- Check that the API_URL in `popup.js` is correct

### No comments fetched

- Video might have comments disabled
- YouTube API quota might be exceeded
- API key might be invalid or expired

## 📝 Development

### File Structure

```
yt-chrome-plugin-frontend/
├── manifest.json      # Chrome extension manifest
├── popup.html         # Extension popup UI
├── popup.js          # Main logic for fetching and displaying data
└── README.md         # This file
```

### Making Changes

1. Edit the files as needed
2. Go to `chrome://extensions/`
3. Click the refresh icon on the extension card
4. Test your changes

## 🌐 API Status

Current API: **http://40.172.234.207:6091**

Check API status:
```bash
curl http://40.172.234.207:6091/health
curl http://40.172.234.207:6091/docs  # API documentation
```

## 📄 License

This extension is part of the YouTube Comment Sentiment Classifier API project.

## 🤝 Support

If you encounter any issues:
1. Check the troubleshooting section above
2. Verify API is running and accessible
3. Check browser console for errors
4. Ensure all permissions are granted

---

**Enjoy analyzing YouTube comments with AI! 🎉**

