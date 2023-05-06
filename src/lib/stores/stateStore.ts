import { writable, type Writable } from 'svelte/store';

export const tabSet: Writable<number> = writable(0);