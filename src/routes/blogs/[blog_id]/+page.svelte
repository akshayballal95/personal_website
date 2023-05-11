<script lang="ts">
	import { onMount } from 'svelte';

	/** @type {import('./$types').PageData} */
	export let data: { id: string; title: string; html: string, description: string };
</script>

<svelte:head>
	<title>{data.title}</title>
	<meta name="description" content="Blogs written by Akshay Ballal" />
	<meta name="keywords" content= {data.description} />
</svelte:head>

<div class="flex justify-center lg:p-10 p-5">
	<div class="flex flex-col gap-8 w-full md:w-3/4 max-w-4xl rounded-md card md:p-8 p-5">
		<h2 class="mt-5">{data.title}</h2>

		{@html data.html}
	</div>
</div>
