import { writable, type Writable } from 'svelte/store';

export const tabSet: Writable<number> = writable(0);
export const projectSet: Writable<number> = writable(0);
export const resumeSet: Writable<number> = writable(0);