
export interface Message {
  id: string;
  text: string;
  sender: 'other' | 'me';
  timestamp: Date;
}

export interface ConversationContext {
  topic: string;
  mood: string;
  summary: string;
}

export enum AppStatus {
  IDLE = 'IDLE',
  CONNECTING = 'CONNECTING',
  LISTENING = 'LISTENING',
  ERROR = 'ERROR'
}
