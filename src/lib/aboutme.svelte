<script>
	import { Avatar } from '@skeletonlabs/skeleton';
	import img from '$lib/assets/img.png';
	let currentVariant = 'bg-initial';
</script>

<div class="bg-transparent h-full w-full flex items-center justify-center">
	<div class=" h-[500px] w-[800px] flex gap-10">
		<div class=" flex flex-col items-center gap-10 card min-w-[300px]">
			<Avatar class="mt-5" src={img} width="w-32" rounded="rounded-full" />
			<h3>Akshay Ballal</h3>
			<h3 class="grow ml-5 mr-5 text-center">CHIEF PRODUCT OFFICER</h3>
			<div
				class=" h-14 w-full bg-white flex text-black justify-around items-center text-3xl self-end"
			>
				<i class="fa-brands fa-twitter" />
				<i class="fa-brands fa-facebook" />
				<i class="fa-brands fa-github" />
			</div>
		</div>
		<div class="flex flex-col justify-start">
			<h2 class = "mb-2">Hello</h2>
			<h3 class="mb-10">Here's who I am and what I do</h3>
            <div class="flex gap-5 mb-10">
                <button type="button" class="btn variant-filled rounded-full">Resume</button>
                <button type="button" class="btn variant-ringed rounded-full">Projects</button>
            </div >
			<p>
				I am passionate about building new technologies that push the boundaries of the status quo.
				My mix of knowledge in both mechanical systems and software development allows me to build
				more holistic systems and products. I have experience in building additive manufacturing
				machines and technologies and have put together several machines from scratch. I am the
				co-inventor of 7 granted patents in the field of composites and additive manufacturing.
			</p>
		</div>
	</div>
</div>
