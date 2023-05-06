<script lang="ts">
	import { TabGroup, Tab } from '@skeletonlabs/skeleton';
	import {tabSet} from "$lib/stores/stateStore";
	import { goto } from '$app/navigation';

	let tabsBottomNav: number = 0;
</script>

<TabGroup
	justify="justify-center"
	active="variant-filled-primary"
	hover="hover:variant-soft-primary"
	flex="flex-1 lg:flex-none"
	rounded=""
	border=""
	class="bg-surface-100-800-token w-full"
>
	<Tab bind:group={$tabSet} on:click={()=>{goto("/")}} name="about_me" value={0} class = "rounded">
		<!-- <svelte:fragment slot="lead">SD</svelte:fragment> -->
		<p class = "font-light">About Me</p>
	</Tab>

	<Tab bind:group={$tabSet} on:click={()=>{goto("/resume")}} name="resume" value={1} class = "rounded">
		<!-- <svelte:fragment slot="lead">SD</svelte:fragment> -->
		<p class = "font-light">Resume</p>
	</Tab>

	<Tab bind:group={$tabSet} name="projects" value={2} class = "rounded">
		<!-- <svelte:fragment slot="lead">SD</svelte:fragment> -->
		<p class = "font-light">Projects</p>
	</Tab>

	<Tab bind:group={$tabSet} name="Contact" value={3} class = "rounded">
		<!-- <svelte:fragment slot="lead">SD</svelte:fragment> -->
		<p class = "font-light">Contact</p>

	</Tab>
	
	<Tab bind:group={$tabSet} name="Blog" value={4} class = "rounded">
		<!-- <svelte:fragment slot="lead">SD</svelte:fragment> -->
		<p class = "font-light">Blog</p>

	</Tab>
	<!-- ... -->
</TabGroup>

