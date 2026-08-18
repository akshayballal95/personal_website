import { error } from '@sveltejs/kit';
import { CHAT_ENABLED } from '$lib/config';

// Guards every route in the (chat) group. With the feature off, /chat 404s
// instead of rendering a UI whose backend refuses to answer.
export function load() {
	if (!CHAT_ENABLED) {
		error(404, 'Not found');
	}
}
