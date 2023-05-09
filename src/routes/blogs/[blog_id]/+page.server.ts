import { Client } from "@notionhq/client"
import { VITE_NOTION_KEY } from "$env/static/private"
import { VITE_DATABASE_ID } from "$env/static/private"
import type { Blog } from "../../../input_model"
import { NotionToMarkdown } from "notion-to-md"
import {compile} from 'mdsvex'

/** @type {import('./$types').PageLoad} */
export async function load({ params }) {

    const notion = new Client({ auth: VITE_NOTION_KEY })
    const n2m = new NotionToMarkdown({ notionClient: notion });


    const mdblocks = await n2m.pageToMarkdown(params.blog_id);
    const mdString = n2m.toMarkdownString(mdblocks);
    const compiled_response = await compile(mdString.parent)
    // console.log(compiled_response);

    return { data: compiled_response?.code }

}