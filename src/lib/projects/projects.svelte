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
</script>

<div class="flex h-full justify-center items-center">
	<div class="bg-transparent w-2/4 flex flex-col gap-8 items-center justify-center p-5">
		<h2>Projects</h2>
		{#each pros as pro}
			<div class="card pt-10 w-full flex flex-col">
				<div class="flex items-center">
					<div class="h-12 w-2 mr-5 bg-primary-500" />

					<p class="unstyled text-2xl text-primary-500">{pro.metadata.title}</p>
				</div>
				<div class="flex ml-8 mt-5">
					<p class="text-justify unstyled text-sm">
						<svelte:component  this = {pro.default}/>
					</p>

					<img
						alt=""
						src="https://static.wixstatic.com/media/a33b49_1067617f8ae1493fb4b0b79e3f355eea~mv2.png/v1/crop/x_1575,y_0,w_3571,h_4480/fill/w_369,h_463,al_c,q_85,usm_0.66_1.00_0.01,enc_auto/5DIV0098.png"
					/>
				</div>
			</div>
		{/each}
	</div>
</div>
