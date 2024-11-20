<script lang="ts">
	import { tabSet } from '$lib/stores/stateStore';
	import { Avatar } from '@skeletonlabs/skeleton';
	import { elemChat, message_feed } from '$lib/stores/messageStore';
	import avatar from '$lib/assets/avatar.jpg'
	import img from '$lib/assets/img.png'
	import { marked } from 'marked';

	$tabSet = 5;



</script>

<svelte:head>
	<title>AI Assistant</title>
	<meta
		name="description"
		content="You can talk to my chatbot assistant about anything related to Machine Learning and Artificial Intelligence."
	/>
</svelte:head>

	<div class="flex flex-col h-full w-full justify-center items-center">

		{#if $message_feed.length!=0}
		<div class="flex flex-col h-full md:w-2/4 gap-5">
			<!-- Conversation -->
			<section bind:this={$elemChat} class=" max-h-fit p-4 overflow-y-scroll space-y-4">
				{#each $message_feed as bubble}
					{#if bubble.host === false}
						<div class="grid grid-cols-[auto_1fr] gap-2">
							<Avatar src={img} width="w-12" />
							<div class="card p-4 variant-soft rounded-tl-none space-y-2">
								<header class="flex justify-between items-center">
									<p class="font-bold">{"Akshay"}</p>
									<small class="opacity-50">{bubble.timestamp}</small>
								</header>
								<p>{@html marked(bubble.message)}</p>
							</div>
						</div>
					{:else}
						<div class="grid grid-cols-[1fr_auto] gap-2">
							<div class="card p-4 rounded-tr-none space-y-2 {bubble.color}">
								<header class="flex justify-between items-center">
									<p class="font-bold">{"You"}</p>
									<small class="opacity-50">{bubble.timestamp}</small>
								</header>
								<p>{@html marked(bubble.message)}</p>
							</div>
							<Avatar src={avatar} width="w-12" />
						</div>
					{/if}
				{/each}
			</section>
		</div>
		{:else}
		<div class = 'flex flex-col items-center gap-3'><h4>Ask me Questions like:</h4>
			<span  class="chip variant-filled">What projects have you worked on</span>
			
			<span class="chip variant-filled">Tell Me about yourself</span>

			<span class="chip variant-filled">Anything about Machine Learning and AI<span>
			</div>
		{/if}
	</div>
<!-- <button class="btn variant-filled" on:click={() => chat('How are you')}>Chat</button> -->
