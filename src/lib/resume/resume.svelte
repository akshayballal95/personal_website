<script lang="ts">
	import { onMount } from 'svelte';
	import type { Resume } from '../../input_model';
	import item from '../../resume.json';
	import { tabSet } from '$lib/stores/stateStore';
	import { fly } from 'svelte/transition';

	let resume = item as Resume;

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
		'Coursera': 'https://upload.wikimedia.org/wikipedia/commons/9/97/Coursera-Logo_600x600.svg',
		'edX': 'https://upload.wikimedia.org/wikipedia/commons/8/8f/EdX.svg',
	};

	// Logos that are self-contained solid tiles (own background) — shown edge-to-edge,
	// without the white padding box used for transparent wordmark logos.
	const solidLogos = new Set(['Coursera']);

	function hideLogo(e: Event) {
		(e.target as HTMLImageElement).closest('.logo-wrapper')?.remove();
	}
</script>

<div class="flex h-full justify-center items-start">
	<div class="bg-transparent w-full flex flex-col lg:w-2/4 gap-12 items-center justify-center p-5">
		<div class="flex flex-col gap-4 items-center">
			<h2 class="text-center">Resume</h2>
			<div class="h-1 w-56 bg-primary-500" />
		</div>

		<!-- Experience -->
		<section class="w-full flex flex-col gap-7">
			<div class="flex items-center gap-3">
				<div class="w-10 h-10 rounded-full flex items-center justify-center text-primary-500 bg-primary-500/15 border border-primary-500/30">
					<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="w-5 h-5">
						<rect x="2" y="7" width="20" height="14" rx="2" />
						<path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16" />
					</svg>
				</div>
				<h3 class="unstyled font-semibold uppercase tracking-wider">Experience</h3>
			</div>

			<div class="flex flex-col gap-8">
				{#each resume.work_experience as experience, i}
					<div class="relative flex gap-5" in:fly={{ duration: 500, y: 60, delay: i * 80 }}>
						<!-- node + connecting line -->
						<div class="relative flex flex-col items-center shrink-0 w-16">
							{#if i < resume.work_experience.length - 1}
								<div class="absolute left-1/2 top-8 -translate-x-1/2 w-0.5 h-[calc(100%+2rem)] bg-primary-500/30" />
							{/if}
							{#if logoMap[experience.company_name]}
								<div class="logo-wrapper z-10 w-16 h-16 flex items-center justify-center rounded-2xl p-1.5 {darkBgLogos.has(experience.company_name) ? 'bg-surface-800' : 'bg-white border border-surface-300-600-token'}">
									<img
										src={logoMap[experience.company_name]}
										alt={experience.company_name}
										class="object-contain max-w-full max-h-full"
										on:error={hideLogo}
									/>
								</div>
							{:else}
								<div class="z-10 w-16 h-16 flex items-center justify-center">
									<div class="w-4 h-4 rounded-full bg-primary-500 ring-4 ring-surface-100-800-token" />
								</div>
							{/if}
						</div>

						<!-- content -->
						<div class="flex flex-col gap-1.5 pt-1 pb-1">
							<p class="unstyled text-sm opacity-60">{experience.start_date} – {experience.end_date}</p>
							<h4 class="unstyled font-semibold uppercase tracking-wide text-primary-500">{experience.job_title}</h4>
							<p class="unstyled text-sm opacity-80">{experience.company_name} · {experience.city}, {experience.country}</p>
							<ul class="list-disc flex flex-col gap-1.5 pl-4 mt-2">
								{#each experience.description.split('\n') as desc}
									<li class="text-sm font-light">{desc}</li>
								{/each}
							</ul>
						</div>
					</div>
				{/each}
			</div>
		</section>

		<!-- Education -->
		<section class="w-full flex flex-col gap-7">
			<div class="flex items-center gap-3">
				<div class="w-10 h-10 rounded-full flex items-center justify-center text-primary-500 bg-primary-500/15 border border-primary-500/30">
					<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="w-5 h-5">
						<path d="M22 10 12 5 2 10l10 5 10-5z" />
						<path d="M6 12v5c0 1 2.5 3 6 3s6-2 6-3v-5" />
						<path d="M22 10v6" />
					</svg>
				</div>
				<h3 class="unstyled font-semibold uppercase tracking-wider">Education</h3>
			</div>

			<div class="flex flex-col gap-8">
				{#each resume.education as education, i}
					<div class="relative flex gap-5" in:fly={{ duration: 500, y: 60, delay: i * 80 }}>
						<!-- node + connecting line -->
						<div class="relative flex flex-col items-center shrink-0 w-16">
							{#if i < resume.education.length - 1}
								<div class="absolute left-1/2 top-8 -translate-x-1/2 w-0.5 h-[calc(100%+2rem)] bg-primary-500/30" />
							{/if}
							{#if logoMap[education.institute_name]}
								<div class="logo-wrapper z-10 w-16 h-16 flex items-center justify-center rounded-2xl p-1.5 {darkBgLogos.has(education.institute_name) ? 'bg-surface-800' : 'bg-white border border-surface-300-600-token'}">
									<img
										src={logoMap[education.institute_name]}
										alt={education.institute_name}
										class="object-contain max-w-full max-h-full"
										on:error={hideLogo}
									/>
								</div>
							{:else}
								<div class="z-10 w-16 h-16 flex items-center justify-center">
									<div class="w-4 h-4 rounded-full bg-primary-500 ring-4 ring-surface-100-800-token" />
								</div>
							{/if}
						</div>

						<!-- content -->
						<div class="flex flex-col gap-1.5 pt-1 pb-1">
							<p class="unstyled text-sm opacity-60">{education.start_date} – {education.end_date}</p>
							<h4 class="unstyled font-semibold uppercase tracking-wide text-primary-500">{education.degree}, {education.field}</h4>
							<p class="unstyled text-sm opacity-80">{education.institute_name} · {education.city}, {education.country}</p>
							<ul class="list-disc flex flex-col gap-1.5 pl-4 mt-2">
								{#each education.description.split('\n') as desc}
									<li class="text-sm font-light">{desc}</li>
								{/each}
							</ul>
						</div>
					</div>
				{/each}
			</div>
		</section>

		<!-- Certificates -->
		<section class="w-full flex flex-col gap-7">
			<div class="flex items-center gap-3">
				<div class="w-10 h-10 rounded-full flex items-center justify-center text-primary-500 bg-primary-500/15 border border-primary-500/30">
					<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="w-5 h-5">
						<circle cx="12" cy="8" r="6" />
						<path d="M8.21 13.89 7 23l5-3 5 3-1.21-9.12" />
					</svg>
				</div>
				<h3 class="unstyled font-semibold uppercase tracking-wider">Certificates</h3>
			</div>

			<div class="card flex flex-col gap-3 p-5" in:fly={{ duration: 500, y: 60 }}>
				{#each resume.certification as certificate, i}
					<div class="flex gap-3 items-center">
						{#if platformLogoMap[certificate.platform]}
							<div class="logo-wrapper w-14 h-14 shrink-0 flex items-center justify-center rounded-xl overflow-hidden {solidLogos.has(certificate.platform) ? '' : 'p-1.5 bg-white border border-surface-300-600-token'}">
								<img
									class="object-contain {solidLogos.has(certificate.platform) ? 'w-full h-full' : 'max-w-full max-h-full'}"
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
					{#if i < resume.certification.length - 1}
						<div class="h-px bg-neutral-600 w-full" />
					{/if}
				{/each}
			</div>
		</section>
	</div>
</div>
