import sendgrid from '@sendgrid/mail';
import { VITE_SENDGRID_API_KEY } from '$env/static/private';
import { toastStore, type ToastSettings } from '@skeletonlabs/skeleton';

/** @type {import('./$types').Actions} */
export const actions = {
    default: async ({ cookies, request }) => {
        // TODO log the user in
        sendgrid.setApiKey(VITE_SENDGRID_API_KEY);

        const data = await request.formData()

        interface FormData {
            firstName: string;
            lastName: string;
            email: string;
            subject: string;
            message: string;
        }
        let formData: FormData = {
            firstName: data.get('firstName') as string,
            lastName: data.get('lastName') as string,
            email: data.get('email') as string,
            subject: data.get('subject') as string,
            message: data.get('message') as string,
        };

        const html = "<p>First Name: " + formData.firstName + "</p>" +
            "<p>Last Name: " + formData.lastName + "</p>" +
            "<p>Email: " + formData.email + "</p>" +
            "<p>Subject: " + formData.subject + "</p>" +
            "<p>Message: " + formData.message + "</p>";

        try {
            const response_send = await sendgrid.send({
                to: 'arballal95@gmail.com',
                from: 'arballal95@protonmail.com',
                subject: formData.subject,
                html: html,
            })


            if (response_send) {

                return { success: "true" };
            }

        }

        catch {

            return { success: "false" };

        }


    }
};