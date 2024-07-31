<script lang="ts">
	import { onMount } from 'svelte';

	/** @type {import('./$types').PageData} */
	export let data: { id: string; date: Date; title: string; html: string; description: string };

	function formattedDate(date: Date) {
		return date
			.toLocaleDateString('en-GB', {
				day: 'numeric',
				month: 'short',
				year: 'numeric'
			})
			.replace(/ /g, ' ');
	}

</script>

<svelte:head>
	<title>{data.title}</title>
	<meta name="description" content={data.description} />
	<meta name="title" content={data.title} />
</svelte:head>

<div class="flex justify-center lg:p-10 p-5">
	<div class="flex flex-col gap-8 w-full h-full md:w-3/4 max-w-4xl rounded-md card md:p-8 p-5 relative">
		<h2 class="mt-5">{data.title}</h2>
		<!-- <p>{formattedDate(data.date)}</p> -->

		{@html data.html}
	</div>
</div>
