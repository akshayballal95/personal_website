<script lang="ts">
	import { onMount } from 'svelte';
	import type { Resume } from '../../input_model';
	import item from '../../resume.json';
	import { tabSet } from '$lib/stores/stateStore';
	let resume = item as Resume;

	export let projects: any;
	let pros: any[];
	$: pros = [];

	onMount(async () => {
		$tabSet = 2;
		projects.projects.forEach(async (element: any) => {
			let project = await import(`../assets/projects/${element.slug}.md`);
			pros = [...pros, project];
		});
	});

	// $: console.log(pros[0].metadata.image)
</script>

<div class="flex h-full justify-center items-center">
	<div class="bg-transparent lg:w-2/4 flex flex-col gap-8 items-center justify-center p-5">
		<h2>Projects</h2>
		{#each pros as pro}
			<div class="card pt-10 pb-10 pr-10 w-full flex flex-col">
				<div class="flex items-center">
					<div class="h-12 w-2 mr-5 bg-primary-500" />

					<p class="unstyled text-2xl text-primary-500">{pro.metadata.title}</p>
				</div>
				<div class="lg:flex-row flex flex-col gap-3 ml-8 mt-5 justify-around ">
					<p class="text-justify unstyled text-sm mr-5 basis-3/5 shrink-0">
						<svelte:component this={pro.default} />
					</p>

					<img alt="" class="object-cover overflow-hidden" src={pro.metadata.image} />
				</div>
			</div>
		{/each}
	</div>
</div>
