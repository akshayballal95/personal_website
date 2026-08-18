const { skeleton } = require('@skeletonlabs/tw-plugin');
const { join } = require('path');

/** @type {import('tailwindcss').Config} */
module.exports = {
	darkMode: 'class',
	content: [
		'./src/**/*.{html,js,svelte,ts}',
		join(require.resolve('@skeletonlabs/skeleton'), '../**/*.{html,js,svelte,ts}')
	],
	theme: {
		extend: {},
	},
	plugins: [
		require('@tailwindcss/forms'),
		require('@tailwindcss/typography'),
		// Skeleton v2 emits its base/component styles through the Tailwind plugin.
		// Theme CSS variables come from src/theme.postcss, so no preset is registered.
		skeleton()
	],
}
