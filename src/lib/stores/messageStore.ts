import { writable, type Writable } from 'svelte/store';
import type { MessageFeed } from '../../messageInterface';

export const message_feed: Writable<MessageFeed[]> = writable([]);
export const elemChat: Writable<HTMLElement> = writable();
export const isLoading: Writable<boolean> = writable(false);