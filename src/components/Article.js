import React from "react"
import { GatsbyImage, getImage } from "gatsby-plugin-image"
import * as styles from "../styles/post-details.module.scss"

import markdownIt from "markdown-it"
// import markdownItKatex from "@iktakahiro/markdown-it-katex"
import Prism from "prismjs"

// customize markdown-it
const options = {
	html: true,
	typographer: true,
	linkify: true,
	highlight: function (str, lang) {
		// var languageString = "language-" + lang
		if (Prism.languages[lang]) {
			return `
                <div class="gatsby-highlight" data-language="${lang}">
                    <pre class="language-${lang} line-numbers"><code class="language-${lang}">${Prism.highlight(str, Prism.languages[lang], lang)}</code></pre>
                </div>
                `
		} else {
			return `
            <div class="gatsby-highlight" data-language="${lang}">
                <pre class="language-${lang} line-numbers"><code class="language-${lang}">${Prism.util.encode(str)}</code></pre>
            </div>
            `
		}
	},
}

const customMarkdownIt = new markdownIt(options)
export default function Article({ html, bodyWidget, title, category, featuredImage, date, publicURL, children }) {
	const featuredImagePath = publicURL
	let bodyRendered = ""
	if (bodyWidget) {
		bodyRendered = customMarkdownIt.render(bodyWidget.props?.value || "")
	}
	html = html || bodyRendered
	return (
		<article className={styles.details}>
			<h1>{title}</h1>
			<div className={styles.metadata}>
				<h3>{new Date(date).toDateString()}</h3>
				<h3>.</h3>
				<h3>{category}</h3>
			</div>

			<div className={styles.featured}>
				{featuredImage?.childImageSharp ? (
					<GatsbyImage image={getImage(featuredImage)} alt={title} />
				) : (
					<img className={styles.alternateImage} src={featuredImagePath} alt={title} />
				)}
			</div>
			<div className={styles.html} dangerouslySetInnerHTML={{ __html: html }}></div>

			{children}
		</article>
	)
}
