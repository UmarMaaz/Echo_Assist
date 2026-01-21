
import * as React from 'react';
import { createClient, SupabaseClient } from '@supabase/supabase-js';
import { AppStatus } from './types.ts';

// Constants
const STORAGE_KEY = 'echoassist_stable_v1';
const CONFIDENCE_THRESHOLD = 0.72;
const CLEAR_DELAY = 1500;

// Supabase Configuration
const SUPABASE_URL = import.meta.env.VITE_SUPABASE_URL;
const SUPABASE_KEY = import.meta.env.VITE_SUPABASE_KEY;

const supabase: SupabaseClient | null = (SUPABASE_URL && SUPABASE_KEY)
  ? createClient(SUPABASE_URL, SUPABASE_KEY)
  : null;

// MediaPipe Globals
declare const Hands: any;
declare const drawConnectors: any;
declare const HAND_CONNECTIONS: any;

enum ViewMode {
  INTERPRETER = 'INTERPRETER',
  TRAINING = 'TRAINING',
  ACADEMY = 'ACADEMY',
  LISTENER = 'LISTENER'
}

interface HandSample {
  id: string;
  normalized: { nx: number; ny: number }[];
  curlStates: number[];
}

interface CustomSign {
  id: string;
  label: string;
  samples: HandSample[];
}

interface Prediction {
  label: string;
  confidence: number;
}

const processHandData = (landmarks: any[]) => {
  const wrist = landmarks[0];
  const palmBase = landmarks[9];
  const palmScale = Math.sqrt(
    Math.pow(palmBase.x - wrist.x, 2) + Math.pow(palmBase.y - wrist.y, 2)
  ) || 0.01;

  const normalized = landmarks.map(l => ({
    nx: (l.x - wrist.x) / palmScale,
    ny: (l.y - wrist.y) / palmScale
  }));

  const tips = [4, 8, 12, 16, 20];
  const pips = [2, 6, 10, 14, 18];

  const curlStates = tips.map((tipIdx, i) => {
    const tip = landmarks[tipIdx];
    const pip = landmarks[pips[i]];
    const tDist = Math.sqrt(Math.pow(tip.x - wrist.x, 2) + Math.pow(tip.y - wrist.y, 2));
    const pDist = Math.sqrt(Math.pow(pip.x - wrist.x, 2) + Math.pow(pip.y - wrist.y, 2));
    return tDist < pDist ? 1 : 0;
  });

  return { normalized, curlStates };
};

// Separate component for reliable canvas updates
const SignPreview = ({ sign }: { sign: CustomSign }) => {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);

  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas && sign.samples[0]) {
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, 250, 250);

        // Draw hand skeleton
        const sample = sign.samples[0];

        // Connections
        const connections = [
          [0, 1], [1, 2], [2, 3], [3, 4], // Thumb
          [0, 5], [5, 6], [6, 7], [7, 8], // Index
          [0, 9], [9, 10], [10, 11], [11, 12], // Middle
          [0, 13], [13, 14], [14, 15], [15, 16], // Ring
          [0, 17], [17, 18], [18, 19], [19, 20] // Pinky
        ];

        ctx.strokeStyle = '#818cf8';
        ctx.lineWidth = 2;
        ctx.lineCap = 'round';

        connections.forEach(([i, j]) => {
          const p1 = sample.normalized[i];
          const p2 = sample.normalized[j];
          if (p1 && p2) {
            ctx.beginPath();
            // Centering: shift Y down by 40px
            ctx.moveTo(125 + p1.nx * 80, 165 + p1.ny * 80);
            ctx.lineTo(125 + p2.nx * 80, 165 + p2.ny * 80);
            ctx.stroke();
          }
        });

        // Landmarks
        sample.normalized.forEach((p: any, i: number) => {
          const isTip = [4, 8, 12, 16, 20].includes(i);
          ctx.beginPath();
          // Centering: shift Y down by 40px
          ctx.arc(125 + p.nx * 80, 165 + p.ny * 80, isTip ? 6 : 4, 0, Math.PI * 2);
          ctx.fillStyle = isTip ? '#f472b6' : '#818cf8';
          ctx.fill();
        });
      }
    }
  }, [sign]); // Re-run when sign changes

  return (
    <canvas
      ref={canvasRef}
      width={250}
      height={250}
      className="mx-auto bg-slate-950 rounded-xl sm:rounded-2xl border border-white/10 max-w-full"
    />
  );
};

export default function App() {
  const [activeMode, setActiveMode] = React.useState<ViewMode>(ViewMode.INTERPRETER);
  const [status, setStatus] = React.useState<AppStatus>(AppStatus.IDLE);
  const [isCloudSynced, setIsCloudSynced] = React.useState<'idle' | 'syncing' | 'success' | 'error'>('idle');
  const [liveTranscript, setLiveTranscript] = React.useState('');
  const [sentence, setSentence] = React.useState<string[]>([]);
  const lastDetectedRef = React.useRef<{ label: string, time: number } | null>(null);
  const audioQueueRef = React.useRef<string[]>([]);
  const isPlayingAudioRef = React.useRef(false);
  const lastSpokenWordRef = React.useRef<string | null>(null);
  const [predictions, setPredictions] = React.useState<Prediction[]>([]);
  const [matchedSign, setMatchedSign] = React.useState<CustomSign | null>(null);
  const [textInput, setTextInput] = React.useState('');
  const [isSpeechActive, setIsSpeechActive] = React.useState(false);

  const videoRef = React.useRef<HTMLVideoElement>(null);
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const audioContextRef = React.useRef<AudioContext | null>(null);
  const handsRef = React.useRef<any>(null);
  const isListeningRef = React.useRef(false);
  const shouldMicKeepRunningRef = React.useRef(false);
  const recognitionRef = React.useRef<any>(null);

  const currentHandDataRef = React.useRef<{ normalized: any[], curlStates: number[] } | null>(null);
  const clearTimerRef = React.useRef<number | null>(null);

  // Sign Library State
  const [customSigns, setCustomSigns] = React.useState<CustomSign[]>(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      return saved ? JSON.parse(saved) : [];
    } catch (e) {
      return [];
    }
  });

  const customSignsRef = React.useRef<CustomSign[]>(customSigns);
  const [teachLabel, setTeachLabel] = React.useState('');
  const [countdown, setCountdown] = React.useState<number | null>(null);

  const syncToSupabase = async (data: CustomSign[]) => {
    if (!supabase) return;
    setIsCloudSynced('syncing');
    try {
      const { error } = await supabase
        .from('echo_library')
        .upsert({
          id: 1,
          payload: data,
          updated_at: new Date().toISOString()
        }, { onConflict: 'id' });

      if (error) throw error;
      setIsCloudSynced('success');
      setTimeout(() => setIsCloudSynced('idle'), 2000);
    } catch (err) {
      console.error("Cloud sync failed:", err);
      setIsCloudSynced('error');
    }
  };

  const loadFromSupabase = async () => {
    if (!supabase) return;
    setIsCloudSynced('syncing');
    try {
      const { data, error } = await supabase
        .from('echo_library')
        .select('payload')
        .eq('id', 1)
        .single();

      if (error && error.code !== 'PGRST116') throw error;
      if (data?.payload && Array.isArray(data.payload)) {
        setCustomSigns(data.payload);
        setIsCloudSynced('success');
        setTimeout(() => setIsCloudSynced('idle'), 2000);
      } else {
        setIsCloudSynced('idle');
      }
    } catch (err) {
      console.error("Cloud fetch failed:", err);
      setIsCloudSynced('error');
    }
  };

  React.useEffect(() => {
    if (supabase) loadFromSupabase();
  }, []);

  React.useEffect(() => {
    customSignsRef.current = customSigns;
    localStorage.setItem(STORAGE_KEY, JSON.stringify(customSigns));

    if (customSigns.length > 0) {
      const timeoutId = setTimeout(() => {
        syncToSupabase(customSigns);
      }, 1200);
      return () => clearTimeout(timeoutId);
    }
  }, [customSigns]);

  React.useEffect(() => {
    if (typeof Hands !== 'undefined') {
      const hands = new Hands({
        locateFile: (file: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
      });
      hands.setOptions({
        maxNumHands: 1,
        modelComplexity: 1,
        minDetectionConfidence: 0.8,
        minTrackingConfidence: 0.8
      });
      hands.onResults(onResults);
      handsRef.current = hands;
    }
    initSpeech();
  }, []);

  const initSpeech = () => {
    const SR = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (SR) {
      const rec = new SR();
      rec.continuous = true;
      rec.interimResults = true;
      rec.lang = 'en-US';
      rec.onstart = () => { setIsSpeechActive(true); console.log('Speech recognition started'); };
      rec.onresult = (e: any) => {
        let text = '';
        for (let i = e.resultIndex; i < e.results.length; ++i) {
          text += e.results[i][0].transcript;
        }
        setLiveTranscript(text);
        console.log('Speech detected:', text);

        // Match ALL spoken words to stored signs (for LISTENER mode)
        const words = text.toUpperCase().trim().split(/\s+/).filter(w => w.length > 0);

        // Check the LAST word spoken for best real-time response
        for (let i = words.length - 1; i >= 0; i--) {
          const word = words[i];
          const found = customSignsRef.current.find(s => s.label === word);
          if (found) {
            console.log('Match found:', word);
            setMatchedSign({ ...found, _ts: Date.now() } as any);
            break;
          }
        }

        // Auto-reset transcript after 2 seconds of silence (faster for rapid speech)
        if (clearTimerRef.current) clearTimeout(clearTimerRef.current);
        clearTimerRef.current = window.setTimeout(() => {
          setLiveTranscript('');
          lastSpokenWordRef.current = null;
          console.log('Transcript cleared');
        }, 2000);
      };

      rec.onerror = (e: any) => {
        console.error('Speech error:', e.error);
        if (e.error !== 'no-speech') setIsSpeechActive(false);
      };

      rec.onend = () => {
        setIsSpeechActive(false);
        // Auto-restart if it was supposed to be running (either Camera or Mic mode)
        if (isListeningRef.current || shouldMicKeepRunningRef.current) {
          console.log('Restarting speech recognition...');
          try { rec.start(); } catch (e) { console.log('Restart ignored'); }
        }
      };
      recognitionRef.current = rec;
    } else {
      console.log('Speech recognition not supported');
    }
  };

  // Eagerly load voices to ensure they are available on mobile
  React.useEffect(() => {
    if ('speechSynthesis' in window) {
      const loadVoices = () => {
        const voices = window.speechSynthesis.getVoices();
        console.log('Voices available:', voices.length);
      };
      loadVoices();
      window.speechSynthesis.onvoiceschanged = loadVoices;
    }
  }, []);

  // Helper to prime audio context for mobile browsers
  // Using a short beep sound (Base64 MP3) is often more reliable than TTS for the first interaction
  // Helper to prime audio context for mobile browsers (The "Gold Standard" fix)
  const unlockAudio = () => {
    try {
      const AudioContext = (window as any).AudioContext || (window as any).webkitAudioContext;
      if (!audioContextRef.current) {
        audioContextRef.current = new AudioContext();
      }

      const ctx = audioContextRef.current;
      if (ctx.state === 'suspended') {
        ctx.resume().then(() => {
          console.log("AudioContext resumed/unlocked");
        });
      }

      // Play a silent oscillator to force the state to 'running'
      const oscillator = ctx.createOscillator();
      const gainNode = ctx.createGain();
      gainNode.gain.value = 0; // Silent
      oscillator.connect(gainNode);
      gainNode.connect(ctx.destination);
      oscillator.start(0);
      oscillator.stop(0.001);

    } catch (e) {
      console.error("AudioContext unlock failed:", e);
    }
  };

  const processAudioQueue = async () => {
    if (audioQueueRef.current.length > 0 && !isPlayingAudioRef.current) {
      isPlayingAudioRef.current = true;
      const audioUrl = audioQueueRef.current.shift();

      if (audioUrl && audioContextRef.current) {
        try {
          const response = await fetch(audioUrl);
          const arrayBuffer = await response.arrayBuffer();
          const audioBuffer = await audioContextRef.current.decodeAudioData(arrayBuffer);

          const source = audioContextRef.current.createBufferSource();
          source.buffer = audioBuffer;
          source.connect(audioContextRef.current.destination);

          source.onended = () => {
            isPlayingAudioRef.current = false;
            URL.revokeObjectURL(audioUrl);
            processAudioQueue();
          };

          source.start(0);
        } catch (e) {
          console.error("Web Audio playback error:", e);
          isPlayingAudioRef.current = false;
          processAudioQueue();
        }
      } else {
        isPlayingAudioRef.current = false;
      }
    }
  };

  // Start speech recognition only (for LISTENER mode)
  const startSpeechOnly = async () => {
    if (shouldMicKeepRunningRef.current) {
      // User wants to stop
      shouldMicKeepRunningRef.current = false;
      recognitionRef.current?.stop();
      setIsSpeechActive(false);
      return;
    }

    // User wants to start
    shouldMicKeepRunningRef.current = true;

    // Unlock audio immediately on user interaction
    unlockAudio();

    try {
      await navigator.mediaDevices.getUserMedia({ audio: true });
      recognitionRef.current?.start();
    } catch (e) {
      console.error('Microphone access denied:', e);
      shouldMicKeepRunningRef.current = false;
      alert('Please allow microphone access for speech recognition.');
    }
  };

  // Function to search for sign by text input
  const searchSign = (query: string) => {
    const q = query.toUpperCase().trim();
    const found = customSigns.find(s => s.label === q);
    setMatchedSign(found || null);
  };

};

const speakWord = async (word: string) => {
  // Prevent repeating the same word consecutively
  if (lastSpokenWordRef.current === word) {
    console.log('Word already spoken recently, skipping:', word);
    return;
  }
  lastSpokenWordRef.current = word;

  const apiKey = import.meta.env.VITE_ELEVENLABS_API_KEY;
  console.log('speakWord called with:', word, 'API key present:', !!apiKey);

  if (!apiKey) {
    console.log('No ElevenLabs API key, falling back to browser TTS');
    fallbackSpeak(word);
    return;
  }

  try {
    const voiceId = 'JBFqnCBsd6RMkjVDRZzb'; // George voice
    // console.log('Calling ElevenLabs API...');
    const response = await fetch(`https://api.elevenlabs.io/v1/text-to-speech/${voiceId}`, {
      method: 'POST',
      headers: {
        'Accept': 'audio/mpeg',
        'Content-Type': 'application/json',
        'xi-api-key': apiKey
      },
      body: JSON.stringify({
        text: word,
        model_id: 'eleven_turbo_v2_5',
        voice_settings: { stability: 0.5, similarity_boost: 0.75 }
      })
    });

    if (!response.ok) {
      throw new Error(`ElevenLabs API error: ${response.status}`);
    }

    const blob = await response.blob();
    const url = URL.createObjectURL(blob);

    // Add to queue and process
    audioQueueRef.current.push(url);
    processAudioQueue();

  } catch (e: any) {
    console.error('TTS Error:', e);
    // Fallback
    fallbackSpeak(word);
  }
};

const fallbackSpeak = (word: string) => {
  if ('speechSynthesis' in window) {
    const utterance = new SpeechSynthesisUtterance(word);
    utterance.rate = 1.0;
    window.speechSynthesis.speak(utterance);
  }
};

const calculateSimilarity = (live: any, saved: any) => {
  let totalDist = 0;
  const weights = [1, 1, 1, 1, 4, 1, 1, 1, 4, 1, 1, 1, 4, 1, 1, 1, 4, 1, 1, 1, 4];
  let weightSum = 0;
  for (let i = 0; i < 21; i++) {
    const dist = Math.sqrt(
      Math.pow(live.normalized[i].nx - saved.normalized[i].nx, 2) +
      Math.pow(live.normalized[i].ny - saved.normalized[i].ny, 2)
    );
    totalDist += dist * weights[i];
    weightSum += weights[i];
  }
  const geomScore = Math.max(0, 1 - (totalDist / weightSum * 2.2));
  let stateMatches = 0;
  for (let i = 0; i < 5; i++) {
    if (live.curlStates[i] === saved.curlStates[i]) stateMatches++;
  }
  const stateScore = stateMatches / 5;
  return (geomScore * 0.7) + (stateScore * 0.3);
};

const onResults = (results: any) => {
  if (!canvasRef.current) return;
  const ctx = canvasRef.current.getContext('2d')!;
  if (videoRef.current) {
    const { videoWidth, videoHeight } = videoRef.current;
    if (canvasRef.current.width !== videoWidth || canvasRef.current.height !== videoHeight) {
      canvasRef.current.width = videoWidth;
      canvasRef.current.height = videoHeight;
    }
  }
  ctx.save();
  ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
  const marks = results.multiHandLandmarks?.[0];

  if (marks) {
    const handData = processHandData(marks);
    currentHandDataRef.current = handData;
    drawConnectors(ctx, marks, HAND_CONNECTIONS, { color: '#6366f1', lineWidth: 4 });

    marks.forEach((point: any, i: number) => {
      const isTip = [4, 8, 12, 16, 20].includes(i);
      const fingerIdx = [4, 8, 12, 16, 20].indexOf(i);
      const color = isTip && handData.curlStates[fingerIdx] === 1 ? '#fb7185' : '#818cf8';
      ctx.beginPath();
      ctx.arc(point.x * canvasRef.current!.width, point.y * canvasRef.current!.height, isTip ? 6 : 3, 0, 2 * Math.PI);
      ctx.fillStyle = color;
      ctx.fill();
    });

    const allMatches: Prediction[] = customSignsRef.current.map(sign => {
      let bestSampleConf = 0;
      sign.samples.forEach(sample => {
        const conf = calculateSimilarity(handData, sample);
        if (conf > bestSampleConf) bestSampleConf = conf;
      });
      return { label: sign.label, confidence: bestSampleConf };
    })
      .filter(p => p.confidence > 0.45)
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, 3);

    if (allMatches.length > 0 && allMatches[0].confidence > CONFIDENCE_THRESHOLD) {
      setPredictions(allMatches);

      const topMatch = allMatches[0];
      const now = Date.now();
      if (!lastDetectedRef.current ||
        (lastDetectedRef.current.label !== topMatch.label && now - lastDetectedRef.current.time > 1000) ||
        (lastDetectedRef.current.label === topMatch.label && now - lastDetectedRef.current.time > 2000)) {

        // Prevent visual duplicates in sentence bar
        setSentence(prev => {
          const last = prev[prev.length - 1];
          return last === topMatch.label ? prev : [...prev, topMatch.label];
        });
        speakWord(topMatch.label);
        lastDetectedRef.current = { label: topMatch.label, time: now };
      }

      if (clearTimerRef.current) clearTimeout(clearTimerRef.current);
      clearTimerRef.current = window.setTimeout(() => {
        setPredictions([]);
      }, CLEAR_DELAY);
    }
  } else {
    currentHandDataRef.current = null;
  }
  ctx.restore();
};

const toggleEngine = async () => {
  if (status === AppStatus.LISTENING) {
    isListeningRef.current = false;
    if (videoRef.current?.srcObject) (videoRef.current.srcObject as MediaStream).getTracks().forEach(t => t.stop());
    setStatus(AppStatus.IDLE);
    return;
  }
  try {
    setStatus(AppStatus.CONNECTING);
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
    if (videoRef.current) {
      videoRef.current.srcObject = stream;
      videoRef.current.onloadedmetadata = () => {
        videoRef.current?.play();
        isListeningRef.current = true;
        setStatus(AppStatus.LISTENING);
        const loop = async () => {
          if (!isListeningRef.current) return;
          if (videoRef.current?.readyState === 4) await handsRef.current.send({ image: videoRef.current });
          requestAnimationFrame(loop);
        };
        loop();
        try { recognitionRef.current?.start(); } catch (e) { }
      };
    }
  } catch (e) { setStatus(AppStatus.ERROR); }
};

const captureVariation = () => {
  if (!teachLabel.trim() || !currentHandDataRef.current || countdown !== null) return;
  setCountdown(3);
  const timer = setInterval(() => {
    setCountdown(prev => {
      if (prev === null || prev <= 1) {
        clearInterval(timer);
        saveSample();
        return null;
      }
      return prev - 1;
    });
  }, 1000);
};

const saveSample = () => {
  const data = currentHandDataRef.current;
  if (!data) return;
  const label = teachLabel.trim().toUpperCase();
  const newSample: HandSample = {
    id: Date.now().toString(),
    normalized: data.normalized,
    curlStates: data.curlStates
  };
  setCustomSigns(prev => {
    const existing = prev.find(s => s.label === label);
    if (existing) {
      return prev.map(s => s.label === label ? { ...s, samples: [...s.samples, newSample] } : s);
    } else {
      return [...prev, { id: Date.now().toString(), label, samples: [newSample] }];
    }
  });
  setTeachLabel('');
};

const clearLibrary = () => {
  if (confirm("This will permanently delete ALL variations from the cloud and this device. Continue?")) {
    setCustomSigns([]);
    localStorage.removeItem(STORAGE_KEY);
    if (supabase) {
      supabase.from('echo_library').delete().eq('id', 1).then(() => {
        setIsCloudSynced('idle');
      });
    });
  }
}

const doesExist = customSigns.some(s => s.label === teachLabel.trim().toUpperCase());

if (!supabase) {
  return (
    <div className="flex items-center justify-center h-screen bg-slate-950 text-white p-4">
      <div className="max-w-md space-y-4 text-center">
        <h1 className="text-3xl font-black text-rose-500">Configuration Missing</h1>
        <p className="text-slate-400">The application requires Supabase credentials to function.</p>
        <div className="bg-slate-900 p-4 rounded-lg text-left text-xs font-mono text-slate-300">
          <p>Please add the following environment variables to your deployment settings (Vercel/GitHub):</p>
          <br />
          <p className="text-indigo-400">VITE_SUPABASE_URL</p>
          <p className="text-indigo-400">VITE_SUPABASE_KEY</p>
        </div>
      </div>
    </div>
  );
}

return (
  <div className="flex flex-col h-screen bg-slate-950 text-slate-100 font-sans overflow-hidden">
    <header className="flex flex-col sm:flex-row items-center justify-between px-4 sm:px-8 py-4 bg-slate-900 border-b border-white/10 shrink-0 gap-4 sm:gap-0">
      <div className="flex items-center gap-3 w-full sm:w-auto">
        <div className="w-10 h-10 bg-indigo-600 rounded-xl flex items-center justify-center shadow-lg shrink-0">
          <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" /></svg>
        </div>
        <div className="flex flex-col">
          <h1 className="text-xl font-black tracking-tighter truncate leading-none">EchoAssist</h1>
          <div className="flex items-center gap-1.5 mt-1">
            <div className={`w-1.5 h-1.5 rounded-full transition-all duration-500 ${isCloudSynced === 'success' ? 'bg-emerald-400 shadow-[0_0_8px_#10b981]' :
              isCloudSynced === 'syncing' ? 'bg-indigo-400 animate-pulse' :
                isCloudSynced === 'error' ? 'bg-rose-500' : 'bg-slate-600'
              }`} />
            <span className="text-[7px] font-black uppercase tracking-widest opacity-40">
              {isCloudSynced === 'success' ? 'Synchronized' : isCloudSynced === 'syncing' ? 'Syncing variations...' : 'Supabase Active'}
            </span>
          </div>
        </div>
      </div>
      <nav className="flex bg-slate-800 p-1 rounded-xl gap-1 w-full sm:w-auto overflow-x-auto">
        {[ViewMode.INTERPRETER, ViewMode.LISTENER, ViewMode.ACADEMY, ViewMode.TRAINING].map(m => (
          <button key={m} onClick={() => setActiveMode(m)} className={`flex-1 sm:flex-none px-3 sm:px-6 py-2 rounded-lg text-[8px] sm:text-[10px] font-bold uppercase tracking-widest transition-all whitespace-nowrap ${activeMode === m ? 'bg-indigo-600 text-white shadow-md' : 'text-slate-400 hover:text-white'}`}>{m}</button>
        ))}
      </nav>
    </header>

    <main className="flex-1 flex flex-col lg:flex-row overflow-hidden relative">
      {activeMode === ViewMode.ACADEMY ? (
        <div className="flex-1 p-4 sm:p-10 overflow-y-auto grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-8 custom-scrollbar">
          <div className="col-span-full mb-4">
            <h2 className="text-2xl font-black">Visual Academy</h2>
            <p className="text-slate-500 text-sm">Educational tools for sign language development.</p>
          </div>
          {[
            { title: 'Data Variations', concept: 'Why recording a sign from 3 angles (left, center, right) improves accuracy.', category: 'Theory' },
            { title: 'JSONB Management', concept: 'How your hand variations are packed into high-speed Supabase blocks.', category: 'Cloud' },
            { title: 'Confidence Curves', concept: 'Understanding the 72% threshold for reliable real-time interpretation.', category: 'AI' }
          ].map((lesson, idx) => (
            <div key={idx} className="p-6 bg-slate-900 rounded-[2rem] border border-white/5 space-y-4 hover:border-indigo-500/50 transition-all cursor-pointer">
              <div className="aspect-video bg-slate-950 rounded-2xl flex items-center justify-center opacity-30 text-indigo-400"><svg className="w-10 h-10" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" /></svg></div>
              <div><span className="text-[8px] font-black uppercase text-indigo-400 tracking-widest">{lesson.category}</span><h3 className="text-lg font-bold">{lesson.title}</h3><p className="text-xs text-slate-500 leading-relaxed mt-1">{lesson.concept}</p></div>
            </div>
          ))}
        </div>
      ) : (
        <>
          <div className="flex-1 relative bg-black flex items-center justify-center overflow-hidden">
            <video ref={videoRef} className="h-full w-full object-cover mirror" playsInline muted />
            <canvas ref={canvasRef} className="absolute inset-0 h-full w-full object-cover mirror pointer-events-none" />
            {countdown && <div className="absolute inset-0 flex items-center justify-center text-[10rem] font-black text-white/40 animate-pulse z-50">{countdown}</div>}

            <div className="absolute top-4 right-4 sm:top-10 sm:right-10 z-20 flex flex-col gap-2 w-48 sm:w-64">
              {predictions.map((p, i) => (
                <div key={p.label} className={`px-4 py-2 sm:px-6 sm:py-3 bg-slate-900/80 backdrop-blur-md rounded-2xl border-2 transition-all duration-300 ${i === 0 ? 'border-indigo-400 scale-105 shadow-2xl' : 'border-white/5 opacity-50'}`}>
                  <div className="flex justify-between items-end mb-1"><p className={`font-black uppercase tracking-tighter ${i === 0 ? 'text-lg sm:text-2xl text-white' : 'text-xs sm:text-sm text-slate-400'}`}>{p.label}</p><p className="text-[10px] font-bold text-indigo-400">{Math.round(p.confidence * 100)}%</p></div>
                  <div className="h-1 bg-white/10 rounded-full overflow-hidden"><div className={`h-full transition-all duration-500 ${i === 0 ? 'bg-indigo-400' : 'bg-slate-500'}`} style={{ width: `${p.confidence * 100}%` }} /></div>
                </div>
              ))}
            </div>

            {activeMode === ViewMode.INTERPRETER && (
              <>
                <div className="absolute bottom-20 sm:bottom-24 inset-x-4 sm:inset-x-10 flex flex-col items-center gap-4">
                  <div className="p-6 sm:p-8 bg-slate-900/70 backdrop-blur-2xl border border-white/10 rounded-[2rem] sm:rounded-[2.5rem] text-center shadow-2xl w-full max-w-4xl">
                    <p className="text-xl sm:text-3xl font-black italic tracking-tighter text-indigo-400 leading-tight mb-2">
                      {sentence.length > 0 ? sentence.join(' ') : <span className="opacity-30">Waiting for signs...</span>}
                    </p>
                  </div>
                  {sentence.length > 0 && (
                    <button onClick={() => { setSentence([]); lastDetectedRef.current = null; }} className="bg-slate-800 text-slate-400 hover:text-white hover:bg-slate-700 px-6 py-2 rounded-full text-xs font-bold uppercase tracking-widest transition-all">Clear Sentence</button>
                  )}
                </div>
              </>
            )}

            {activeMode === ViewMode.LISTENER && (
              <div className="absolute inset-0 flex flex-col items-center justify-center p-4 sm:p-8 overflow-y-auto">
                <div className="bg-slate-900/90 backdrop-blur-xl border border-white/10 rounded-[1.5rem] sm:rounded-[2rem] p-4 sm:p-8 max-w-2xl w-full text-center space-y-4 sm:space-y-6">
                  <div>
                    <p className="text-[9px] sm:text-[10px] font-black uppercase tracking-widest text-indigo-400 mb-2">Speech-to-Sign Mode</p>
                    <p className="text-lg sm:text-2xl lg:text-4xl font-black text-white leading-tight">
                      {liveTranscript || <span className="opacity-30">Activate microphone...</span>}
                    </p>
                  </div>

                  <button
                    onClick={startSpeechOnly}
                    className={`px-8 py-4 rounded-2xl font-black text-sm tracking-widest transition-all ${isSpeechActive ? 'bg-rose-500 animate-pulse' : 'bg-emerald-600 hover:bg-emerald-500'}`}
                  >
                    {isSpeechActive ? '🎤 LISTENING...' : '🎤 START MICROPHONE'}
                  </button>

                  <div className={`text-xs font-bold uppercase tracking-widest ${isSpeechActive ? 'text-emerald-400' : 'text-slate-500'}`}>
                    {isSpeechActive ? '● Speech Recognition Active' : '○ Microphone Off'}
                  </div>

                  <div className="flex flex-col sm:flex-row gap-2">
                    <input
                      type="text"
                      value={textInput}
                      onChange={(e) => setTextInput(e.target.value)}
                      onKeyDown={(e) => { if (e.key === 'Enter') searchSign(textInput); }}
                      placeholder="Type a word..."
                      className="flex-1 bg-slate-950 border border-white/10 rounded-xl px-4 py-3 text-base sm:text-lg font-bold outline-none focus:border-indigo-500"
                    />
                    <button
                      onClick={() => searchSign(textInput)}
                      className="px-6 py-3 bg-indigo-600 hover:bg-indigo-500 rounded-xl font-bold uppercase text-xs sm:text-sm tracking-wide"
                    >
                      Search
                    </button>
                  </div>

                  {matchedSign ? (
                    <div className="space-y-3 sm:space-y-4">
                      <p className="text-xs sm:text-sm text-emerald-400 font-bold uppercase tracking-widest">Sign Found: {matchedSign.label}</p>
                      <SignPreview sign={matchedSign} />
                    </div>
                  ) : (
                    <div className="py-12 text-slate-500 text-sm">
                      <p>Speak a word that matches a trained sign</p>
                      <p className="text-xs mt-2 opacity-50">Available: {customSigns.map(s => s.label).join(', ') || 'None trained yet'}</p>
                    </div>
                  )}
                </div>
              </div>
            )}

            <button onClick={toggleEngine} className={`absolute bottom-6 sm:bottom-8 left-1/2 -translate-x-1/2 px-8 py-4 sm:px-12 sm:py-6 rounded-2xl font-black text-xs sm:text-sm tracking-widest shadow-2xl transition-all z-30 whitespace-nowrap ${status === AppStatus.LISTENING ? 'bg-rose-500' : 'bg-indigo-600 hover:bg-indigo-500'}`}>{status === AppStatus.LISTENING ? 'DISABLE ENGINE' : 'ACTIVATE CAMERA'}</button>
          </div>

          {activeMode === ViewMode.TRAINING && (
            <aside className="w-full lg:w-[380px] bg-slate-900 border-t lg:border-t-0 lg:border-l border-white/10 flex flex-col shrink-0 overflow-hidden h-[50vh] lg:h-auto">
              <div className="p-4 sm:p-8 space-y-4 sm:space-y-6 bg-slate-900/50">
                <div className="flex justify-between items-center">
                  <h2 className="text-[10px] font-black text-indigo-400 uppercase tracking-[0.4em]">Variation Manager</h2>
                  <button onClick={clearLibrary} className="p-2 bg-slate-800 rounded-lg hover:bg-rose-900 transition-colors"><svg className="w-4 h-4 text-rose-500" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" /></svg></button>
                </div>
                <div className="space-y-4">
                  <div className="flex flex-col gap-1">
                    <label className="text-[10px] font-black opacity-40 uppercase ml-1">Gesture Name</label>
                    <input value={teachLabel} onChange={e => setTeachLabel(e.target.value)} placeholder="e.g. HELLO" className="w-full bg-slate-950 border border-white/10 rounded-xl px-5 py-4 text-lg font-bold outline-none focus:border-indigo-500" />
                  </div>
                  <button onClick={captureVariation} disabled={!teachLabel.trim() || status !== AppStatus.LISTENING || countdown !== null} className={`w-full py-5 rounded-xl font-black uppercase tracking-widest transition-all ${doesExist ? 'bg-amber-600 hover:bg-amber-500 shadow-amber-900/20' : 'bg-indigo-600 hover:bg-indigo-500 shadow-indigo-900/20'} disabled:opacity-20`}>{countdown !== null ? `Capturing...` : doesExist ? 'Add Variation' : 'Train New'}</button>
                </div>
              </div>

              <div className="flex-1 overflow-y-auto px-4 sm:px-8 pb-10 space-y-3 custom-scrollbar">
                <div className="flex items-center justify-between pt-4 border-t border-white/5">
                  <h3 className="text-[10px] font-black opacity-30 uppercase tracking-widest">Variation Library</h3>
                  <div className="flex items-center gap-1">
                    <div className={`w-1 h-1 rounded-full ${isCloudSynced === 'success' ? 'bg-emerald-500' : 'bg-slate-600'}`} />
                    <span className="text-[7px] font-black opacity-40 uppercase">Safe Cloud</span>
                  </div>
                </div>
                {customSigns.length === 0 ? (
                  <div className="py-12 text-center text-[10px] font-bold opacity-20 uppercase tracking-widest border border-dashed border-white/10 rounded-2xl">Empty Library</div>
                ) : (
                  customSigns.slice().reverse().map(sign => (
                    <div key={sign.id} className="p-4 bg-slate-800/40 rounded-2xl border border-white/5 flex items-center justify-between group hover:bg-slate-800/60 transition-all">
                      <div className="overflow-hidden">
                        <div className="flex items-center gap-2">
                          <p className="font-bold text-base truncate">{sign.label}</p>
                          <svg className="w-3 h-3 text-emerald-500" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" /></svg>
                        </div>
                        <div className="flex gap-1 items-center mt-1">
                          <div className="flex gap-1">
                            {sign.samples.map((_, i) => (
                              <div key={i} className="w-1.5 h-1.5 rounded-full bg-indigo-500" />
                            ))}
                          </div>
                          <span className="text-[8px] font-black opacity-30 uppercase ml-1">{sign.samples.length} Recorded</span>
                        </div>
                      </div>
                      <button onClick={() => setCustomSigns(prev => prev.filter(s => s.id !== sign.id))} className="p-2 text-slate-600 hover:text-rose-400 transition-all opacity-0 group-hover:opacity-100"><svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" /></svg></button>
                    </div>
                  ))
                )}
              </div>
            </aside>
          )}
        </>
      )}
    </main>
  </div>
);
