<script lang="ts">
	import { TabGroup, Tab } from '@skeletonlabs/skeleton';
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
	<Tab bind:group={tabsBottomNav} name="about_me" value={0} class = "rounded">
		<!-- <svelte:fragment slot="lead">SD</svelte:fragment> -->
		<p class = "font-thin">About Me</p>
	</Tab>

	<Tab bind:group={tabsBottomNav} name="resume" value={1} class = "rounded">
		<!-- <svelte:fragment slot="lead">SD</svelte:fragment> -->
		<p class = "font-thin">Resume</p>
	</Tab>

	<Tab bind:group={tabsBottomNav} name="projects" value={2} class = "rounded">
		<!-- <svelte:fragment slot="lead">SD</svelte:fragment> -->
		<p class = "font-thin">Projects</p>
	</Tab>

	<Tab bind:group={tabsBottomNav} name="Contact" value={3} class = "rounded">
		<!-- <svelte:fragment slot="lead">SD</svelte:fragment> -->
		<p class = "font-thin">Contact</p>

	</Tab>
	
	<Tab bind:group={tabsBottomNav} name="Blog" value={4} class = "rounded">
		<!-- <svelte:fragment slot="lead">SD</svelte:fragment> -->
		<p class = "font-thin">Blog</p>

	</Tab>
	<!-- ... -->
</TabGroup>

