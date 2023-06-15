<script>
	import { Avatar } from '@skeletonlabs/skeleton';
	import img from '$lib/assets/img.png';
	import resume from '$lib/assets/projects/resume.pdf';
	import { goto } from '$app/navigation';
</script>

<div
	class=" md:flex-row flex flex-col gap-5 md:p-0 items-center bg-neutral-950 bg-opacity-0"
>

	<div class=" flex flex-col items-center gap-5 card md:min-w-[350px] h-full w-full md:w-auto justify-center">
		<Avatar class="mt-10 mb-5" src={img} width="w-40" rounded="rounded-full" />
		<h1 class="unstyled text-4xl">Akshay Ballal</h1>
		<div class="h-0.5 mt-5 w-14 bg-primary-500 rounded-full" />
		<p class="unstyled grow ml-5 mr-5 text-center font-light text-lg tracking-widest">
			MACHINE LEARNING ENTHUSIAST
		</p>
		<div
			class=" h-14 w-full bg-primary-50 flex text-surface-700 justify-around items-center text-3xl self-end"
		>
			<a class="unstyled" href="https://twitter.com/akshayballal95"
				><i class="fa-brands fa-twitter" /></a
			>
			<a class="unstyled" href="https://www.facebook.com/arballal/"
				><i class="fa-brands fa-facebook" /></a
			>
			<a class="unstyled" href="https://github.com/akshayballal95"
				><i class="fa-brands fa-github" /></a
			>
			<a class="unstyled" href="https://www.linkedin.com/in/akshay-ballal/"
				><i class="fa-brands fa-linkedin" /></a
			>

		</div>

	</div>

	<div class="flex flex-col justify-center h-full">
		<h2 class="mb-2">Hello</h2>
		<h3 class="mb-10">Here's who I am and what I do</h3>
		<div class="flex flex-col lg:flex-row gap-5 mb-10">
			<a href={resume} download
				><button type="button" class="btn variant-filled rounded-full w-full"
					><i class="fa-regular fa-download" /> <span>Resume</span></button
				></a
			>
			<button
				on:click={() => goto('/chat')}
				type="button"
				class="btn  variant-filled-surface rounded-full">Chat With My AI Assistant</button
			>
		</div>
		<p class="text-justify">
			Welcome to my personal website! I am a self-taught AI developer driven by a passion for
			pushing the boundaries of technology. Applying First Principles thinking, I strive to solve
			complex challenges and create innovative solutions. As a Technology Enthusiast, I constantly
			explore the latest advancements in the field. I am deeply committed to leveraging AI for
			social good and advocate for green technology. Join me on this journey as I utilize my
			self-taught expertise to build products and technologies that address industrial problems.
			Let's collaborate and shape a better future together.
		</p>
	</div>
</div>
