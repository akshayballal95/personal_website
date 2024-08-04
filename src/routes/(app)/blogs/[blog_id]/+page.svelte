<script lang="ts">
	import { tabSet } from '$lib/stores/stateStore';
	import { onMount } from 'svelte';

	/** @type {import('./$types').PageData} */
	export let data: { id: string; date: Date; title: string; html: string; description: string; image_url: string };

	function formattedDate(date: Date) {
		return date
			.toLocaleDateString('en-GB', {
				day: 'numeric',
				month: 'short',
				year: 'numeric'
			})
			.replace(/ /g, ' ');
	}
	let y = 0;
	onMount(() => {
		$tabSet = 4;
		document.body.querySelector('#blog')?.scrollTo(0, 0);
	});
</script>

<svelte:head>
	<title>{data.title}</title>
	<meta name="description" content={data.description} />
	<meta name="title" content={data.title} />
	<meta
		property="og:image"
		content={data.image_url}
	/>
	<meta
		property="og:description"
		content={data.description}/>
	<meta property="og:title" content={data.title}/>
	<meta property="og:description" content={data.description} />
	

	<meta name="twitter:card" content="summary_large_image" />
	<meta name="twitter:creator" content="@akshayballal95" />
	<meta property="twitter:image"
		content={data.image_url} />
	<meta property="twitter:title" content={data.title} />
	<meta property="twitter:description"
		content={data.description}/>

		<meta name="robots" content="follow, index, max-snippet:-1, max-video-preview:-1, max-image-preview:large" />
		</svelte:head
>

<svelte:window bind:scrollY={y} />

<div class="flex justify-center lg:p-10 p-5 mx-auto overflow-hidden">
	<div id="blog" class="flex flex-col gap-8 md:w-3/4 max-w-4xl rounded-md card md:p-8 p-5 relative">
		<h2 class="mt-5">{data.title}</h2>
		<p>{formattedDate(data.date)}</p>

		{@html data.html}
	</div>
</div>
