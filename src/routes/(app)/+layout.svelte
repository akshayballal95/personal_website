<script lang="ts">
	// import '../theme.postcss';
	// The ordering of these imports is critical to your app working properly
	import '../../theme.postcss';
	// import '@skeletonlabs/skeleton/themes/theme-wintry.css';
	// Most of your app wide CSS should be put in this file
	import '../../app.postcss';

	import { AppShell, Drawer, ListBox, ListBoxItem, Toast, LightSwitch, getDrawerStore } from '@skeletonlabs/skeleton';
	import Header from '$lib/header.svelte';
	import Footer from '$lib/footer.svelte';
	import { goto, afterNavigate } from '$app/navigation';
	import { tabSet } from '$lib/stores/stateStore';
	import { inject } from '@vercel/analytics';
	import { dev } from '$app/environment';

	const drawer = getDrawerStore();

	inject({ mode: dev ? 'development' : 'production' });
	
	afterNavigate(() => {
		// Target the main content area of AppShell
		document.querySelector('#page')?.scrollTo(0, 0);
	});
</script>


<AppShell
>
	<svelte:fragment slot="header"><Header /></svelte:fragment>

	<!-- Router Slot -->
	<slot />
	<!-- ---- / ---- -->
	<svelte:fragment slot="pageFooter"><Footer/></svelte:fragment>

	<Drawer  padding="0" position="top" regionBackdrop="w-full" regionDrawer="h-auto" >
		<ListBox
			hover="hover:variant-soft-primary"
			active="variant-filled-primary"
			class=" pt-6 gap-4 flex flex-col justify-center items-center  "
		>	
			<button class="btn-icon variant-filled absolute self-end top-5 right-5" on:click={()=>drawer.close()}
				><i class="fa-solid fa-xmark fa-xl"></i>
			</button>
			<ListBoxItem
				bind:group={$tabSet}
				on:click={() => {
					goto('/');
					drawer.close();
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
					drawer.close();
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
					drawer.close();
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
					drawer.close();
				}}
			>
				<!-- <svelte:fragment slot="lead">SD</svelte:fragment> -->
				<p class="font-light">Contact</p>
			</ListBoxItem>

			<ListBoxItem bind:group={$tabSet} name="Blog" value={4} class="rounded"
			on:click={() => {
				goto('/blogs');
				drawer.close();
			}}>
				<!-- <svelte:fragment slot="lead">SD</svelte:fragment> -->
				<p class="font-light">Blog</p>
			</ListBoxItem>

			<ListBoxItem bind:group={$tabSet} name="Chat" value={5} class="rounded"
			on:click={() => {
				goto('/chat');
				drawer.close();
			}}>
				<!-- <svelte:fragment slot="lead">SD</svelte:fragment> -->
				<p class="font-light">Chat</p>
			</ListBoxItem>
			<!-- ... -->
		</ListBox>
		<div class="flex justify-center pb-6 pt-2">
			<LightSwitch />
		</div>
	</Drawer>
</AppShell>

<Toast />

