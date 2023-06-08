<script lang="ts">
	// import '../theme.postcss';
	// The ordering of these imports is critical to your app working properly
	import '@skeletonlabs/skeleton/themes/theme-gold-nouveau.css';
	// If you have source.organizeImports set to true in VSCode, then it will auto change this ordering
	import '@skeletonlabs/skeleton/styles/all.css';
	// Most of your app wide CSS should be put in this file
	import '../../app.postcss';

	import {
		AppShell,
		Drawer,
		ListBox,
		ListBoxItem,
		Toast,
		drawerStore
	} from '@skeletonlabs/skeleton';
	import Header from '$lib/header.svelte';
	import { goto } from '$app/navigation';
	import { tabSet } from '$lib/stores/stateStore';
	import { inject } from '@vercel/analytics';
	import { dev } from '$app/environment';
	import { onMount } from 'svelte';
	import { message_feed, elemChat } from '$lib/stores/messageStore';
	inject({ mode: dev ? 'development' : 'production' });

	function scrollChatBottom(behavior?: ScrollBehavior): void {
		$elemChat.scrollTo({ top: $elemChat.scrollHeight, behavior });
	}

	let currentMessage = '';
	$: currentMessage
	function addMessage(): void {
		const newMessage = {
			id: Date.now(),
			host: true,
			avatar: 48,
			name: 'Jane',
			timestamp: getCurrentTimestamp(),
			message: currentMessage,
			color: 'variant-soft-primary'
		};
		// Append the new message to the message feed
		$message_feed = [...$message_feed, newMessage];
		// Clear the textarea message
        	// Smoothly scroll to the bottom of the feed
		setTimeout(() => {
			scrollChatBottom('smooth');
		}, 0);
        chat(currentMessage)
        	// Smoothly scroll to the bottom of the feed
		setTimeout(() => {
			scrollChatBottom('smooth');
		}, 0);

		currentMessage = '';
	
	}

	function keyHandler(e:KeyboardEvent){
		if (e.key=="Enter"){
			addMessage()
			currentMessage = ''
		}
	}

	function getCurrentTimestamp(): string {
		return new Date().toLocaleString('en-US', { hour: 'numeric', minute: 'numeric', hour12: true });
	}
	onMount(() => scrollChatBottom('smooth'));

	async function chat(message: String) {
		const response = await fetch('api/chat',  {
			method: 'POST',
			
			headers: {
				'Content-Type': 'application/json',
				'Access-Control-Allow-Origin': '*',
			},
			body: JSON.stringify({
				human_input: message
			})
		});
	

		let AImessage =await response.json()
        const newMessage = {
			id: Date.now(),
			host: false,
			avatar: 48,
			name: 'Jane',
			timestamp: getCurrentTimestamp(),
			message:AImessage.output,
			color: 'variant-soft-primary'
		};
		// Append the new message to the message feed
		$message_feed = [...$message_feed, newMessage];
		setTimeout(() => {
			scrollChatBottom('smooth');
		}, 0);
	}


</script>

<AppShell
	regionPage="relative"
	slotPageFooter=" sticky bottom-0 z-10 flex w-full justify-center"
	slotPageContent="overflow-y-scroll"
>
	<svelte:fragment slot="header"><Header /></svelte:fragment>

	<!-- Router Slot -->
	<slot />
	<!-- ---- / ---- -->
	<svelte:fragment slot="pageFooter">
		<div
			class="m-5 input-group input-group-divider grid-cols-[auto_1fr_auto] rounded-container-token md:w-2/4"
		>
			<div class="input-group-shim"></div>
			<textarea
			on:keydown={(e)=>keyHandler(e)}
				bind:value={currentMessage}
				class="bg-transparent border-0 ring-0"
				name="prompt"
				id="prompt"
				placeholder="Write a message..."
				rows="1"
			/>
			<button  on:click={addMessage} class="variant-filled-primary">Send</button>
		</div>
	</svelte:fragment>

	<Drawer padding="0" position="top" regionBackdrop="w-full">
		<ListBox
			hover="hover:variant-soft-primary"
			active="variant-filled-primary"
			class=" m-6 gap-5 flex flex-col justify-center items-center"
		>
			<button
				class="btn-icon variant-filled absolute self-end top-5 right-5"
				on:click={() => drawerStore.close()}
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

			<ListBoxItem
				bind:group={$tabSet}
				name="Blog"
				value={4}
				class="rounded"
				on:click={() => {
					goto('/blogs');
					drawerStore.close();
				}}
			>
				<!-- <svelte:fragment slot="lead">SD</svelte:fragment> -->
				<p class="font-light">Blog</p>
			</ListBoxItem>
			<!-- ... -->
		</ListBox>
	</Drawer>
</AppShell>

<Toast />
