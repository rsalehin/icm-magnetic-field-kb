// Background service worker — minimal, just keeps extension alive
chrome.runtime.onInstalled.addListener(() => {
  console.log('ICM Knowledge Base extension installed');
});