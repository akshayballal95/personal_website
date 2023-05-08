<script lang="ts">
	import { onMount } from 'svelte';
	import type { Resume } from '../../input_model';
	import item from '../../resume.json';
	import { tabSet } from '$lib/stores/stateStore';

	let resume = item as Resume;

    onMount(()=>{
        $tabSet = 1;
    })
</script>
<div class="flex h-full justify-center items-center">
<div class="bg-transparent lg:w-2/4 w-full  flex flex-col gap-8 items-center justify-center p-5">
    <h2> Resume </h2>

    <h2 class = "self-start">Experience</h2>
	{#each resume.work_experience as experience}
		<div class="card w-full p-5 gap-3 flex flex-col lg:items-center lg:flex-row">
			<div class="flex flex-col gap-2 basis-1/4 shrink-0">
				<p class="unstyled text-2xl mb-3">{experience.start_date} - {experience.end_date}</p>
				<p class="unstyled font-semibold text-primary-500">{experience.job_title}</p>
				<p class="unstyled text-sm">{experience.company_name}</p>
				<p class="unstyled text-sm">{experience.city}, {experience.country}</p>
			</div>
			<div class=" lg:h-60 lg:w-1 w-60 h-0.5 mr-6 bg-primary-500 " />

			<ul class="list-disc flex flex-col text-justify gap-3 pl-3 lg:pl-0">
				{#each experience.description.split('\n') as desc}
					<li>
						<dt class="font-light text-sm">{desc}</dt>
					</li>
				{/each}
			</ul>
		</div>
	{/each}

    <h2 class = "self-start">Education</h2>
	{#each resume.education as education}
		<div class="card w-full p-5 gap-3 flex flex-col lg:items-center lg:flex-row">
			<div class="flex flex-col gap-2 basis-1/4 shrink-0">
				<p class="unstyled text-2xl mb-3">{education.start_date} - {education.end_date}</p>
				<p class="unstyled font-semibold text-primary-500">{education.institute_name}</p>
				<p class="unstyled">{education.degree}</p>
                <p class="unstyled">{education.field}</p>
				<p class="unstyled">{education.city}, {education.country}</p>
			</div>
			<div class=" lg:h-60 lg:w-1 w-60 h-0.5 mr-6 bg-primary-500 " />

			<ul class="list-disc flex flex-col text-justify gap-3 pl-3 lg:pl-0">
				{#each education.description.split('\n') as desc}
					<li>
						<dt class="font-light text-sm">{desc}</dt>
					</li>
				{/each}
			</ul>
		</div>
	{/each}
</div>
</div>
