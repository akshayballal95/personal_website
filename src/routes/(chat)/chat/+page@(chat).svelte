<script lang="ts">
	import { tabSet } from '$lib/stores/stateStore';
	import { Avatar } from '@skeletonlabs/skeleton';
	import { elemChat, message_feed, isLoading } from '$lib/stores/messageStore';
	import avatar from '$lib/assets/avatar.jpg'
	import img from '$lib/assets/img.png'
	import { marked } from 'marked';
	import DOMPurify from 'dompurify';

	function safeMarkdown(text: string): string {
		return DOMPurify.sanitize(marked(text) as string);
	}

	$tabSet = 5;

	const suggestions = [
		"Tell me about EmbedAnything",
		"What's your current role?",
		"How did you build Asia's first carbon fiber 3D printer?",
		"What did you work on at ASML?",
		"What's your educational background?",
		"What are your core ML skills?",
	];
</script>

<svelte:head>
	<title>AI Assistant</title>
	<meta
		name="description"
		content="Chat with Akshay's AI assistant about his experience, projects, and research in Machine Learning and AI."
	/>
</svelte:head>

<div class="flex flex-col h-full w-full justify-center items-center">

	{#if $message_feed.length !== 0}
		<div class="flex flex-col h-full w-full md:w-2/4 gap-5">
			<section bind:this={$elemChat} class="max-h-fit p-4 overflow-y-auto space-y-4">
				{#each $message_feed as bubble}
					{#if bubble.host === false}
						<div class="grid grid-cols-[auto_1fr] gap-2">
							<Avatar src={img} width="w-10" />
							<div class="card p-4 variant-soft rounded-tl-none space-y-2">
								<header class="flex justify-between items-center">
									<p class="font-bold text-sm">Akshay</p>
									<small class="opacity-50 text-xs">{bubble.timestamp}</small>
								</header>
								<div class="text-sm prose dark:prose-invert max-w-none">{@html safeMarkdown(bubble.message)}</div>
							</div>
						</div>
					{:else}
						<div class="grid grid-cols-[1fr_auto] gap-2">
							<div class="card p-4 rounded-tr-none space-y-2 {bubble.color}">
								<header class="flex justify-between items-center">
									<p class="font-bold text-sm">You</p>
									<small class="opacity-50 text-xs">{bubble.timestamp}</small>
								</header>
								<p class="text-sm">{bubble.message}</p>
							</div>
							<Avatar src={avatar} width="w-10" />
						</div>
					{/if}
				{/each}

				{#if $isLoading}
					<div class="grid grid-cols-[auto_1fr] gap-2">
						<Avatar src={img} width="w-10" />
						<div class="card p-4 variant-soft rounded-tl-none w-fit">
							<div class="flex gap-1 items-center h-5 px-1">
								<span class="typing-dot" />
								<span class="typing-dot" style="animation-delay: 0.2s" />
								<span class="typing-dot" style="animation-delay: 0.4s" />
							</div>
						</div>
					</div>
				{/if}
			</section>
		</div>
	{:else}
		<div class="flex flex-col items-center gap-6 px-4 text-center max-w-md">
			<Avatar src={img} width="w-20" />
			<div class="flex flex-col gap-1">
				<h3 class="unstyled font-bold text-xl">Hey, I'm Akshay</h3>
				<p class="unstyled text-sm opacity-60">ML Engineer · Open Source · Rust & Python</p>
			</div>
			<p class="unstyled text-sm opacity-70">Ask me anything about my work, projects, or background.</p>
			<div class="flex flex-wrap gap-2 justify-center">
				{#each suggestions as s}
					<span class="chip variant-soft hover:variant-filled cursor-pointer text-xs">{s}</span>
				{/each}
			</div>
		</div>
	{/if}
</div>

<style>
	.typing-dot {
		width: 7px;
		height: 7px;
		border-radius: 9999px;
		background-color: currentColor;
		opacity: 0.5;
		animation: typing-bounce 0.8s ease-in-out infinite;
	}

	@keyframes typing-bounce {
		0%, 80%, 100% { transform: translateY(0); opacity: 0.4; }
		40% { transform: translateY(-5px); opacity: 1; }
	}
</style>
