<script lang="ts">
	import { onMount } from 'svelte';
	import type { Resume } from '../../input_model';
	import item from '../../resume.json';
	import { projectSet, tabSet } from '$lib/stores/stateStore';
	import { Tab, TabGroup } from '@skeletonlabs/skeleton';
	import { goto } from '$app/navigation';

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
		<TabGroup
			justify="justify-center"
			active="variant-filled-primary"
			hover="hover:variant-soft-primary"
			flex="flex-1 lg:flex-none"
			rounded=""
			border=""
			class="bg-surface-100-800-token rounded "
		>
			<Tab bind:group={$projectSet} name="about_me" value={0} class="rounded">
				<!-- <svelte:fragment slot="lead">SD</svelte:fragment> -->
				<p class="font-light">Software</p>
			</Tab>

			<Tab bind:group={$projectSet} name="resume" value={1} class="rounded">
				<!-- <svelte:fragment slot="lead">SD</svelte:fragment> -->
				<p class="font-light">Hardware</p>
			</Tab>
		</TabGroup>
		{#each pros as pro}
			{#if (pro.metadata.type == 'software' && $projectSet == 0) || (pro.metadata.type == 'hardware' && $projectSet == 1) }
				<div class="card pt-10 pb-10 pr-10 w-full flex flex-col">
					<div class="flex items-center">
						<div class="h-12 w-2 mr-5 bg-primary-500" />

						<p class="unstyled text-2xl text-primary-500">{pro.metadata.title}</p>
					</div>
					<div class="lg:flex-row flex flex-col gap-5 ml-8 mt-5 justify-around items-center">
						<p class="text-justify unstyled text-sm mr-5 basis-3/5 shrink-0">
							<svelte:component this={pro.default} />
						</p>
						<div class=" lg:h-80 lg:w-1 w-80 h-0.5 mr-6 bg-primary-500 " />

						<div class = " w-72">
							<img
								alt=""
								class=" object-contain"
					
								src={pro.metadata.image}
							/>
						</div>
					</div>
				</div>

			{/if}
		{/each}
	</div>
</div>
