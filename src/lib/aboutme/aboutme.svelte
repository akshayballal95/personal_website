<script>
	import { Avatar } from '@skeletonlabs/skeleton';
	import img from '$lib/assets/img.png';
	let currentVariant = 'bg-initial';
</script>

<div class="bg-transparent h-full w-full flex items-center justify-center">
	<div class=" h-[500px] w-[800px] flex gap-10">
		<div class=" flex flex-col items-center gap-5 card min-w-[350px]">
			<Avatar class="mt-10 mb-5" src={img} width="w-40" rounded="rounded-full" />
			<h3>Akshay Ballal</h3>
			<div class="h-0.5 mt-5 w-14 bg-[#e6c833] rounded-full" />
			<p class="unstyled grow ml-5 mr-5 text-center font-light text-lg tracking-widest">
				CHIEF PRODUCT OFFICER
			</p>
			<div
				class=" h-14 w-full bg-primary-50 flex text-black justify-around items-center text-3xl self-end"
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
		<div class="flex flex-col justify-start">
			<h2 class="mb-2">Hello</h2>
			<h3 class="mb-10">Here's who I am and what I do</h3>
			<div class="flex gap-5 mb-10">
				<button type="button" class="btn variant-filled rounded-full">Resume</button>
				<button type="button" class="btn variant-ringed rounded-full">Projects</button>
			</div>
			<p class="text-justify">
				I am passionate about building new technologies that push the boundaries of the status quo.
				My mix of knowledge in both mechanical systems and software development allows me to build
				more holistic systems and products. I have experience in building additive manufacturing
				machines and technologies and have put together several machines from scratch. I am the
				co-inventor of 7 granted patents in the field of composites and additive manufacturing.
			</p>
		</div>
	</div>
</div>
