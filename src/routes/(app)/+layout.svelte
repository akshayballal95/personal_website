<script lang="ts">
	// import '../theme.postcss';
	// The ordering of these imports is critical to your app working properly
	import '@skeletonlabs/skeleton/themes/theme-gold-nouveau.css';
	// If you have source.organizeImports set to true in VSCode, then it will auto change this ordering
	import '@skeletonlabs/skeleton/styles/all.css';
	// Most of your app wide CSS should be put in this file
	import '../../app.postcss';

	import { AppShell, Drawer, ListBox, ListBoxItem, Toast, drawerStore } from '@skeletonlabs/skeleton';
	import Header from '$lib/header.svelte';
	import Footer from '$lib/footer.svelte';
	import { goto } from '$app/navigation';
	import { tabSet } from '$lib/stores/stateStore';
	import { inject } from '@vercel/analytics';
	import { dev } from '$app/environment';
	inject({ mode: dev ? 'development' : 'production' });
	
</script>

<AppShell>
	<svelte:fragment slot="header"><Header /></svelte:fragment>

	<!-- Router Slot -->
	<slot />
	<!-- ---- / ---- -->

	<Drawer padding="0" position="top" regionBackdrop="w-full">
		<ListBox
			hover="hover:variant-soft-primary"
			active="variant-filled-primary"
			class=" m-6 gap-5 flex flex-col justify-center items-center"
		>
			<button class="btn-icon variant-filled absolute self-end top-5 right-5" on:click={()=>drawerStore.close()}
				><i class="fa-solid fa-xmark fa-xl" />
			</button>
			<ListBoxItem
				bind:group={$tabSet}
				on:click={() => {
					goto('/');
					drawerStore.close();
				}}
				name="about_me"
				value={0}
				class="rounded"
			>
				<!-- <svelte:fragment slot="lead">SD</svelte:fragment> -->
				<p class="font-light">About Me</p>
			</ListBoxItem>

			<ListBoxItem
				bind:group={$tabSet}
				on:click={() => {
					goto('/resume');
					drawerStore.close();
				}}
				name="resume"
				value={1}
				class="rounded"
			>
				<!-- <svelte:fragment slot="lead">SD</svelte:fragment> -->
				<p class="font-light">Resume</p>
			</ListBoxItem>

			<ListBoxItem
				bind:group={$tabSet}
				on:click={() => {
					goto('/projects');
					drawerStore.close();
				}}
				name="projects"
				value={2}
				class="rounded"
			>
				<!-- <svelte:fragment slot="lead">SD</svelte:fragment> -->
				<p class="font-light">Projects</p>
			</ListBoxItem>

			<ListBoxItem
				bind:group={$tabSet}
				name="Contact"
				value={3}
				class="rounded"
				on:click={() => {
					goto('/contact');
					drawerStore.close();
				}}
			>
				<!-- <svelte:fragment slot="lead">SD</svelte:fragment> -->
				<p class="font-light">Contact</p>
			</ListBoxItem>

			<ListBoxItem bind:group={$tabSet} name="Blog" value={4} class="rounded"
			on:click={() => {
				goto('/blogs');
				drawerStore.close();
			}}>
				<!-- <svelte:fragment slot="lead">SD</svelte:fragment> -->
				<p class="font-light">Blog</p>
			</ListBoxItem>
			<!-- ... -->
		</ListBox>
	</Drawer>
</AppShell>

<Toast />

