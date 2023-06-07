<script lang="ts">
	import { TabGroup, Tab, ListBox, ListBoxItem } from '@skeletonlabs/skeleton';
	import { tabSet } from '$lib/stores/stateStore';
	import { goto } from '$app/navigation';

	import { Drawer, drawerStore } from '@skeletonlabs/skeleton';
	import type { DrawerSettings } from '@skeletonlabs/skeleton';

	let innerWidth = 0;

	function trigger(position: 'left' | 'top' | 'right' | 'bottom'): void {
		const s: DrawerSettings = { id: 'demo', position };
		drawerStore.open(s);
	}
</script>

<svelte:window bind:innerWidth />

<TabGroup
	justify="justify-center gap-2"
	active="variant-filled-primary"
	hover="hover:variant-soft-primary"
	flex="flex-1 lg:flex-none"
	rounded=""
	border=""
	class="bg-surface-100-800-token w-full hidden md:block"
>
	<Tab
		bind:group={$tabSet}
		on:click={() => {
			goto('/');
		}}
		name="about_me"
		value={0}
		class="rounded"
	>
		<!-- <svelte:fragment slot="lead">SD</svelte:fragment> -->
		<p class="font-light">About Me</p>
	</Tab>

	<Tab
		bind:group={$tabSet}
		on:click={() => {
			goto('/resume');
		}}
		name="resume"
		value={1}
		class="rounded"
	>
		<!-- <svelte:fragment slot="lead">SD</svelte:fragment> -->
		<p class="font-light">Resume</p>
	</Tab>

	<Tab
		bind:group={$tabSet}
		on:click={() => {
			goto('/projects');
		}}
		name="projects"
		value={2}
		class="rounded"
	>
		<!-- <svelte:fragment slot="lead">SD</svelte:fragment> -->
		<p class="font-light">Projects</p>
	</Tab>

	<Tab
		bind:group={$tabSet}
		name="Contact"
		value={3}
		class="rounded"
		on:click={() => {
			goto('/contact');
		}}
	>
		<!-- <svelte:fragment slot="lead">SD</svelte:fragment> -->
		<p class="font-light">Contact</p>
	</Tab>

	<Tab
		bind:group={$tabSet}
		name="Blog"
		value={4}
		class="rounded"
		on:click={() => {
			goto('/blogs');
		}}
	>
		<!-- <svelte:fragment slot="lead">SD</svelte:fragment> -->
		<p class="font-light">Blog</p>
	</Tab>

	<Tab
	bind:group={$tabSet}
	name="Chat"
	value={5}
	class="rounded"
	on:click={() => {
		goto('/chat');
	}}
>
	<!-- <svelte:fragment slot="lead">SD</svelte:fragment> -->
	<p class="font-light">Chat</p>
</Tab>


	

	<!-- ... -->
</TabGroup>

<button
	class="btn-icon variant-filled md:hidden"
	on:click={() => {
		trigger('top');
	}}
>
	<i class="fa-solid fa-arrow-down" />
</button>
