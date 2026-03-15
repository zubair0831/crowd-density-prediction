'use strict';
// ════════════════════════════════════════════════════════════════════════════
// speech.js — Automated crowd announcement system
// ════════════════════════════════════════════════════════════════════════════

const SpeechSystem = {
  enabled: true, speaking: false, lastLevel: null, lastAnnounceTime: 0,
  COOLDOWN_MS: 30000,  // minimum 30 s between any two announcements
  logEntries: [],

  messages: {
    moderate: [
      "Attention: Crowd density is increasing. Please walk slowly and follow designated routes.",
      "Notice: This area is becoming busy. Please maintain spacing and proceed calmly.",
    ],
    high: [
      "Warning: This area is becoming crowded. Please move toward alternative routes and avoid stopping.",
      "Caution: Significant crowd density detected. Please disperse to nearby open areas when possible.",
    ],
    critical: [
      "Important safety announcement: This area is overcrowded. Please remain calm, stop moving forward, and follow instructions.",
      "Safety alert: Critical crowd density detected. Please cooperate with staff and move toward exit routes immediately.",
    ],
    predictive: [
      "Crowd levels are rising in this area. Please move gradually and use alternate pathways.",
      "Notice: Crowd density is trending upward. Please begin moving to less congested areas as a precaution.",
    ],
    normalizing: [
      "The situation is improving. Thank you for your patience and cooperation.",
    ],
  },

  _pick(arr) { return arr[Math.floor(Math.random() * arr.length)]; },

  announce(level, custom) {
    if (!this.enabled || !('speechSynthesis' in window)) return;
    const now = Date.now();
    if (now - this.lastAnnounceTime < this.COOLDOWN_MS && level !== 'critical') return;
    if (level === this.lastLevel && level !== 'critical') return;

    const msg = custom || this._pick(this.messages[level] || this.messages.moderate);
    window.speechSynthesis.cancel();
    const utt = new SpeechSynthesisUtterance(msg);
    utt.rate = 0.88; utt.pitch = 1.0; utt.volume = 1.0;
    const voices = window.speechSynthesis.getVoices();
    const pref = voices.find(v => v.lang.startsWith('en') && v.name.includes('Female'))
              || voices.find(v => v.lang.startsWith('en'))
              || voices[0];
    if (pref) utt.voice = pref;

    utt.onstart = () => {
      this.speaking = true;
      document.getElementById('speakDot').classList.add('active');
      document.getElementById('speechStatus').classList.add('speaking');
      document.getElementById('speechStatusText').textContent = msg;
    };
    utt.onend = () => {
      this.speaking = false;
      document.getElementById('speakDot').classList.remove('active');
      document.getElementById('speechStatus').classList.remove('speaking');
      document.getElementById('speechStatusText').textContent = 'System ready — automated warnings enabled';
    };
    window.speechSynthesis.speak(utt);
    this.lastAnnounceTime = now;
    this.lastLevel = level;
    this._log(level, msg);
  },

  _log(level, msg) {
    const entry = { time: new Date().toLocaleTimeString(), level, msg };
    this.logEntries.unshift(entry);
    if (this.logEntries.length > 20) this.logEntries.pop();
    this._renderLog();
    notableEvents.unshift({ ...entry, type: 'announcement', frame: frameData.length });
  },

  _renderLog() {
    const el = document.getElementById('announceLog');
    if (!this.logEntries.length) return;
    el.innerHTML = this.logEntries.map(e =>
      `<div class="al-entry">
        <div class="al-entry-time">${e.time}</div>
        <span class="al-entry-sev sev-${e.level}">${e.level.toUpperCase()}</span>
        <div class="al-entry-msg">${e.msg}</div>
      </div>`
    ).join('');
  },

  evaluate(fd, threshold) {
    if (!this.enabled || !fd) return;
    const ratio   = fd.count / Math.max(threshold, 1);
    const state   = fd.behavior?.state;
    const trend   = fd.behavior?.count_trend || 0;
    const modPct  = (+document.getElementById('threshMod').value  || 75)  / 100;  // was 60
    const hiPct   = (+document.getElementById('threshHigh').value || 90)  / 100;  // was 80
    const crPct   = (+document.getElementById('threshCrit').value || 100) / 100;
    const predPPS = +document.getElementById('threshPred').value  || 5;            // was 3

    if (state === 'SURGE' || ratio >= crPct)           this.announce('critical');
    else if (state === 'PRE_SURGE' || ratio >= hiPct)  this.announce('high');
    else if (ratio >= modPct)                          this.announce('moderate');
    else if (trend >= predPPS && ratio > 0.35)         this.announce('predictive');
    else if (state === 'NORMAL' && this.lastLevel && this.lastLevel !== 'normalizing') {
      if (Date.now() - this.lastAnnounceTime > 25000) {
        this.announce('normalizing');
        this.lastLevel = null;
      }
    }
  },
};

function toggleSpeech() {
  SpeechSystem.enabled = !SpeechSystem.enabled;
  const btn = document.getElementById('speechToggle');
  btn.textContent = SpeechSystem.enabled ? 'Mute' : 'Unmute';
  btn.classList.toggle('muted', !SpeechSystem.enabled);
  if (!SpeechSystem.enabled) window.speechSynthesis.cancel();
  document.getElementById('speechStatusText').textContent = SpeechSystem.enabled
    ? 'System ready — automated warnings enabled'
    : 'Announcements muted';
}

function testAnnouncement() {
  SpeechSystem.lastAnnounceTime = 0;
  SpeechSystem.lastLevel = null;
  SpeechSystem.announce('moderate', 'This is a test of the automated crowd warning system. System is operating normally.');
}