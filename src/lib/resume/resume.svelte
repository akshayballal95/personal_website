<script lang="ts">
	import { onMount } from 'svelte';
	import type { Certification, Resume } from '../../input_model';
	import item from '../../resume.json';
	import { resumeSet, tabSet } from '$lib/stores/stateStore';
	import { fly } from 'svelte/transition';
	import { Tab, TabGroup } from '@skeletonlabs/skeleton';

	let resume = item as Resume;

	$: innerWidth = 2000;

	onMount(() => {
		$tabSet = 1;
	});

	const logoMap: Record<string, string> = {
		'ASML': 'https://upload.wikimedia.org/wikipedia/commons/6/6c/ASML_Holding_N.V._logo.svg',
		'ASML (via TMC)': 'https://upload.wikimedia.org/wikipedia/commons/6/6c/ASML_Holding_N.V._logo.svg',
		'ChatLicense (Freelancer)': 'https://chatlicense.com/wp-content/themes/chatlicense/assets/images/chatlicence-logo.png',
		'Fabheads Automation': 'https://fabheads.com/assets/images/header/logo.png',
		'Eindhoven University of Technology': 'https://upload.wikimedia.org/wikipedia/commons/6/67/Eindhoven_University_of_Technology_logo_new.svg',
		'Birla Institute of Technology, Mesra': 'https://upload.wikimedia.org/wikipedia/en/d/d2/Birla_Institute_of_Technology_Mesra.png',
	};

	// Logos with light/white artwork that need a dark background to be visible in light theme
	const darkBgLogos = new Set(['Fabheads Automation']);

	const platformLogoMap: Record<string, string> = {
		'Coursera': 'https://upload.wikimedia.org/wikipedia/commons/thumb/9/97/Coursera-Logo_600x600.svg/900px-Coursera-Logo_600x600.svg.png',
		'edX': 'https://upload.wikimedia.org/wikipedia/commons/thumb/8/8f/EdX.svg/320px-EdX.svg.png',
	};

	function hideLogo(e: Event) {
		(e.target as HTMLImageElement).closest('.logo-wrapper')?.remove();
	}
</script>

<svelte:window bind:innerWidth />

<div class="flex h-full justify-center items-start">
	<div class="bg-transparent w-full flex flex-col lg:w-2/4 gap-8 items-center justify-center p-5">
		<div class=" flex flex-col gap-4 items-center">
			<h2 class=" text-center">Resume</h2>
			<div class="h-1 w-56 bg-primary-500" />
			<TabGroup
				justify="justify-center"
				active="variant-filled-primary"
				hover="hover:variant-soft-primary"
				flex="flex-1 lg:flex-none"
				rounded=""
				border=""
				class="bg-surface-100-800-token rounded"
			>
				<Tab bind:group={$resumeSet} name="experience" value={0} class="rounded">
					<!-- <svelte:fragment slot="lead">SD</svelte:fragment> -->
					<p class="unstyled text-sm">Experience</p>
				</Tab>

				<Tab bind:group={$resumeSet} name="education" value={1} class="rounded">
					<!-- <svelte:fragment slot="lead">SD</svelte:fragment> -->
					<p class="unstyled text-sm">Education</p>
				</Tab>
				<Tab bind:group={$resumeSet} name="certificates" value={2} class="rounded">
					<!-- <svelte:fragment slot="lead">SD</svelte:fragment> -->
					<p class="unstyled text-sm">Certificates</p>
				</Tab>
			</TabGroup>
		</div>

		{#if $resumeSet == 0}
			{#each resume.work_experience as experience}
				<div
					class="card w-full p-5 gap-3 flex flex-col lg:items-center lg:flex-row"
					in:fly={{ duration: 500, y: 150 }}
				>
					<div class="flex flex-col gap-2 basis-1/4 shrink-0">
						{#if logoMap[experience.company_name]}
							<div class="logo-wrapper w-32 h-14 flex items-center justify-center rounded-lg mb-2 px-3 {darkBgLogos.has(experience.company_name) ? 'bg-surface-800' : 'bg-surface-100-800-token border border-surface-300-600-token'}">
								<img
									src={logoMap[experience.company_name]}
									alt={experience.company_name}
									class="object-contain max-w-full max-h-full"
									on:error={hideLogo}
								/>
							</div>
						{/if}
						<p class="unstyled text-2xl mb-3">{experience.start_date} - {experience.end_date}</p>
						<p class="unstyled font-semibold text-primary-500">{experience.job_title}</p>
						<p class="unstyled text-sm">{experience.company_name}</p>
						<p class="unstyled text-sm">{experience.city}, {experience.country}</p>
					</div>
					<div class=" lg:h-60 lg:w-1 w-60 h-0.5 mr-6 bg-primary-500" />

					<ul class="list-disc flex flex-col text-justify gap-3 pl-3 lg:block lg:pl-0">
						{#each experience.description.split('\n') as desc}
							<li>
								<dt class="font-light text-sm">{desc}</dt>
							</li>
						{/each}
					</ul>
				</div>
			{/each}
		{/if}
		{#if $resumeSet == 1}
			{#each resume.education as education}
				<div
					class="card w-full p-5 gap-3 flex flex-col lg:items-center lg:flex-row"
					in:fly={{ duration: 500, y: 150 }}
				>
					<div class="flex flex-col gap-2 basis-1/4 shrink-0">
						{#if logoMap[education.institute_name]}
							<div class="logo-wrapper w-32 h-14 flex items-center justify-center rounded-lg mb-2 px-3 bg-surface-100-800-token border border-surface-300-600-token">
								<img
									src={logoMap[education.institute_name]}
									alt={education.institute_name}
									class="object-contain max-w-full max-h-full"
									on:error={hideLogo}
								/>
							</div>
						{/if}
						<p class="unstyled text-2xl mb-3">{education.start_date} - {education.end_date}</p>
						<p class="unstyled font-semibold text-primary-500">{education.institute_name}</p>
						<p class="unstyled">{education.degree}</p>
						<p class="unstyled">{education.field}</p>
						<p class="unstyled">{education.city}, {education.country}</p>
					</div>
					<div class=" lg:h-60 lg:w-1 w-60 h-0.5 mr-6 bg-primary-500" />

					<ul class="list-disc flex flex-col text-justify gap-3 pl-3 lg:pl-0">
						{#each education.description.split('\n') as desc}
							<li>
								<dt class="font-light text-sm">{desc}</dt>
							</li>
						{/each}
					</ul>
				</div>
			{/each}
		{/if}

		{#if $resumeSet == 2}
			<!-- <h3 class="self-start col-start-2 row-start-4 hidden lg:block">Certificates</h3> -->

			<div
				class="card flex flex-col gap-3 divide-neutral-500 p-5 " in:fly={{ duration: 500, y: 150 }} out:fly={{ duration: 500, y: -150 }} 
			>
				{#each resume.certification as certificate}
					<div class="flex gap-3 items-center">
						{#if platformLogoMap[certificate.platform]}
							<div class="logo-wrapper w-20 h-14 shrink-0 flex items-center justify-center rounded-lg px-2 bg-surface-100-800-token border border-surface-300-600-token">
								<img
									class="object-contain max-w-full max-h-full"
									alt={certificate.platform}
									src={platformLogoMap[certificate.platform]}
									on:error={hideLogo}
								/>
							</div>
						{/if}
						<div class="flex flex-col items-start gap-1">
							<p class="unstyled font-bold text-sm">{certificate.course_name}</p>
							<p class="unstyled text-xs">{certificate.platform}</p>

							<p class="unstyled text-xs text-neutral-400">Issued {certificate.date}</p>
							<p class="unstyled text-xs text-neutral-400">{certificate.credential}</p>
						</div>
					</div>
					<div class="h-px bg-neutral-600 w-full" />
				{/each}
			</div>
		{/if}
	</div>
</div>
