// Content script — runs on Claude, ChatGPT, Gemini
// Checks if we were opened by ICM-KB and injects the prompt

(async function () {
  const params   = new URLSearchParams(window.location.search);
  const promptId = params.get('kb_prompt_id');
  if (!promptId) return;

  // Clean URL so prompt_id doesn't stay visible
  const cleanUrl = window.location.origin + window.location.pathname;
  window.history.replaceState({}, '', cleanUrl);

  // Fetch prompt from local KB server
  let promptText;
  try {
    const res  = await fetch(`http://localhost:8000/api/pending-prompt/${promptId}`);
    const data = await res.json();
    promptText = data.prompt;
  } catch (e) {
    console.warn('ICM-KB: could not fetch prompt', e);
    return;
  }

  if (!promptText) return;

  // Wait for the input to appear then inject
  await waitForInput(promptText);
})();


async function waitForInput(text, maxWait = 15000) {
  const start    = Date.now();
  const hostname = window.location.hostname;

  while (Date.now() - start < maxWait) {
    const injected = tryInject(hostname, text);
    if (injected) return;
    await sleep(500);
  }
  console.warn('ICM-KB: timed out waiting for input');
}


function tryInject(hostname, text) {
  if (hostname.includes('claude.ai'))    return injectClaude(text);
  if (hostname.includes('chatgpt.com'))  return injectChatGPT(text);
  if (hostname.includes('gemini.google')) return injectGemini(text);
  return false;
}


function injectClaude(text) {
  // Try multiple Claude selectors
  const input =
    document.querySelector('div[contenteditable="true"][data-placeholder]') ||
    document.querySelector('.ProseMirror') ||
    document.querySelector('div[contenteditable="true"]');

  if (!input) return false;

  input.focus();

  // Method 1 — paste event
  const dt = new DataTransfer();
  dt.setData('text/plain', text);
  input.dispatchEvent(new ClipboardEvent('paste', {
    clipboardData: dt,
    bubbles: true,
    cancelable: true,
  }));

  setTimeout(() => {
    // Check if paste worked
    if (input.textContent.trim().length < 10) {
      // Method 2 — execCommand
      input.textContent = '';
      input.focus();
      document.execCommand('insertText', false, text);
    }

    // Find and click send
    setTimeout(() => {
      const btn =
        document.querySelector('button[aria-label="Send Message"]') ||
        document.querySelector('button[aria-label="Send message"]') ||
        document.querySelector('button[data-testid="send-button"]') ||
        document.querySelector('button[type="submit"]');

      if (btn && !btn.disabled) {
        btn.click();
      }
    }, 600);
  }, 400);

  return true;
}


function injectChatGPT(text) {
  // Try multiple selectors — ChatGPT changes these frequently
  const input = 
    document.querySelector('#prompt-textarea') ||
    document.querySelector('div[contenteditable="true"]') ||
    document.querySelector('textarea');

  if (!input) return false;

  input.focus();

  if (input.tagName === 'TEXTAREA') {
    // Standard textarea
    const nativeSetter = Object.getOwnPropertyDescriptor(
      window.HTMLTextAreaElement.prototype, 'value'
    ).set;
    nativeSetter.call(input, text);
    input.dispatchEvent(new Event('input', { bubbles: true }));
  } else {
    // ContentEditable div (newer ChatGPT)
    input.textContent = '';
    const dt = new DataTransfer();
    dt.setData('text/plain', text);
    input.dispatchEvent(new ClipboardEvent('paste', {
      clipboardData: dt,
      bubbles: true,
      cancelable: true,
    }));
  }

  // Wait longer for ChatGPT's React to process
  setTimeout(() => {
    // Try multiple send button selectors
    const btn =
      document.querySelector('button[data-testid="send-button"]') ||
      document.querySelector('button[aria-label="Send prompt"]') ||
      document.querySelector('button[aria-label="Send message"]') ||
      document.querySelector('button[class*="send"]');

    if (btn && !btn.disabled) {
      btn.click();
    }
  }, 800);

  return true;
}


function injectGemini(text) {
  // Gemini uses a rich text editor
  const input = document.querySelector(
    'div[contenteditable="true"], rich-textarea div[contenteditable]'
  );
  if (!input) return false;

  input.focus();
  const dt = new DataTransfer();
  dt.setData('text/plain', text);
  input.dispatchEvent(new ClipboardEvent('paste', {
    clipboardData: dt,
    bubbles: true,
    cancelable: true,
  }));

  setTimeout(() => {
    if (input.textContent.trim() === '') {
      document.execCommand('insertText', false, text);
    }
    setTimeout(() => clickSend('button[aria-label*="Send"], mat-icon[data-mat-icon-name*="send"]'), 600);
  }, 400);

  return true;
}


function clickSend(selector) {
  const btn = document.querySelector(selector);
  if (btn && !btn.disabled) {
    btn.click();
  } else {
    // Fallback: press Enter
    document.activeElement?.dispatchEvent(
      new KeyboardEvent('keydown', { key: 'Enter', bubbles: true })
    );
  }
}


function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}