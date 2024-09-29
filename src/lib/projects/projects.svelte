<script lang="ts">
	import { onMount } from 'svelte';
	import type { Resume } from '../../input_model';
	import item from '../../resume.json';
	import { projectSet, tabSet } from '$lib/stores/stateStore';
	import { Tab, TabGroup } from '@skeletonlabs/skeleton';
	import { goto } from '$app/navigation';
	import { fly, slide } from 'svelte/transition';
	import { parse } from 'postcss';

	let resume = item as Resume;

	// export let projects: any;
	let pros: any[];
	$: pros = [];

	onMount(async () => {
		$tabSet = 2;

		const paths = import.meta.glob('/src/lib/assets/projects/*.md', { eager: true });

		for (const path in paths) {
			const file: any = paths[path];
			pros = [...pros, file];	
		}
		console.log(pros[2].metadata)

	});
</script>


<div class="flex h-full justify-center items-center p-5">
	<div class="md:w-3/4 lg:w-2/4 lg:min-w-[800px] flex flex-col gap-8 items-center justify-center">
		<div class=" flex flex-col gap-4 items-center">
			<h2>Projects</h2>
			<div class="h-1 w-56 bg-primary-500" />

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
					<p class="unstyled text-sm">Software</p>
				</Tab>

				<Tab bind:group={$projectSet} name="resume" value={1} class="rounded">
					<!-- <svelte:fragment slot="lead">SD</svelte:fragment> -->
					<p class="unstyled text-sm">Hardware</p>
				</Tab>
			</TabGroup>
		</div>
		<!-- <div >
			<iframe class = "lg:w-[1200px] lg:h-[700px] w-full h-[800px] p-0"
				title="CarVision"
				src="https://akshayballal-carvision.hf.space"
				frameborder="0"
			/>
		</div> -->

		{#each pros.sort((a,b)=> a.metadata.idx - b.metadata.idx) as pro}
			{#if (pro.metadata.type == 'software' && $projectSet == 0) || (pro.metadata.type == 'hardware' && $projectSet == 1)}
				<div
					class="card pt-10 pb-10 md:pr-10 w-full flex flex-col"
					in:fly={{ duration: 500, x: $projectSet == 0 ? -150 : 150 }}
				>
					<div class="flex items-center">
						<div class="h-12 w-2 mr-5 bg-primary-500" />

						<p class="unstyled text-2xl text-primary-500">{pro.metadata.title}</p>
					</div>
					<div class="lg:flex-row flex flex-col gap-5 mt-5 justify-around items-center">
						<p
							class="text-justify unstyled text-sm ml-5 mr-5 lg:mr-5 lg:ml-8 md:basis-3/5 shrink-0"
						>
							<svelte:component this={pro.default} />
						</p>
						<div class=" lg:h-80 lg:w-1 w-64 h-0.5 mr-6 bg-primary-500" />

						<div class=" w-72">
							<img alt="" class=" object-contain rounded-md" src={pro.metadata.image} />
						</div>
					</div>
				</div>
			{/if}
		{/each}
	</div>
</div>
