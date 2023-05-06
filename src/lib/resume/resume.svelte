<script lang="ts">
	import type { Resume } from '../../input_model';
	import item from '../../resume.json';

	let resume = item as Resume;
</script>
<div class="flex h-full justify-center items-center">
<div class="bg-transparent w-2/4  flex flex-col gap-8 items-center justify-center p-5">
    <h2> Resume </h2>

    <h2 class = "self-start">Experience</h2>
	{#each resume.work_experience as experience}
		<div class="card p-5 flex items-center">
			<div class="flex flex-col gap-2 basis-1/4 shrink-0">
				<p class="unstyled text-2xl mb-3">{experience.start_date} - {experience.end_date}</p>
				<p class="unstyled font-semibold text-primary-500">{experience.job_title}</p>
				<p class="unstyled text-sm">{experience.company_name}</p>
				<p class="unstyled text-sm">{experience.city}, {experience.country}</p>
			</div>
			<div class="h-full w-px mr-6 bg-[#e6c833] rounded-full" />

			<ul class="list-disc flex flex-col gap-3">
				{#each experience.description.split('\n') as desc}
					<li>
						<dt class="font-light text-sm">{desc}</dt>
					</li>
				{/each}
			</ul>
		</div>
	{/each}

    <h2 class = "self-start">Education</h2>
	{#each resume.work_experience as experience}
		<div class="card p-5 flex items-center">
			<div class="flex flex-col gap-2 basis-1/4 shrink-0">
				<p class="unstyled text-2xl mb-3">{experience.start_date} - {experience.end_date}</p>
				<p class="unstyled font-semibold text-primary-500">{experience.job_title}</p>
				<p class="unstyled text-sm">{experience.company_name}</p>
				<p class="unstyled text-sm">{experience.city}, {experience.country}</p>
			</div>
			<div class="h-full w-px mr-6 bg-[#e6c833] rounded-full" />

			<ul class="list-disc flex flex-col gap-3">
				{#each experience.description.split('\n') as desc}
					<li>
						<dt class="font-light text-sm">{desc}</dt>
					</li>
				{/each}
			</ul>
		</div>
	{/each}
</div>
</div>
