import React from "react"
import PropTypes from "prop-types"
import Article from "../../components/Article"

// import Prism from "prismjs"
// // customize markdown-it
// const options = {
// 	html: true,
// 	typographer: true,
// 	linkify: true,
// 	highlight: function (str, lang) {
// 		// var languageString = "language-" + lang
// 		if (Prism.languages[lang]) {
// 			return `<pre class="language-${lang} line-numbers"><code class="language-${lang}">${Prism.highlight(str, Prism.languages[lang], lang)}</code></pre>`
// 		} else {
// 			return `<pre class="language-${lang} line-numbers"><code class="language-${lang}">${Prism.util.encode(str)}</code></pre>`
// 		}
// 	},
// }

// const customMarkdownIt = new markdownIt(options)

const PostDetailsPreview = ({ entry, widgetFor, getAsset }) => {
	//  id, html, excerpt, title, category, featuredImage, date, publicURL
	const title = entry.getIn(["data", "title"])
	const date = entry.getIn(["data", "date"])
	const category = entry.getIn(["data", "category"])
	const featuredImage = entry.getIn(["data", "featuredImage"])
	console.log("featuredImage:", featuredImage)
	const publicURL = featuredImage ? getAsset(featuredImage)?.toString() : null
	const bodyWidget = widgetFor("body")
	// let bodyRendered = ""
	// if (bodyWidget) {
	// 	bodyRendered = customMarkdownIt.render(bodyWidget.props?.value || "")
	// }
	return (
		<Article
			{...{
				// id: entry.getIn(["data", "id"]),
				// excerpt: entry.getIn(["data", "excerpt"]),
				title,
				category,
				featuredImage,
				date,
				publicURL,
				bodyWidget,
			}}
		></Article>
	)
}

PostDetailsPreview.propTypes = {
	entry: PropTypes.shape({
		getIn: PropTypes.func,
	}),
	widgetFor: PropTypes.func,
}
export default PostDetailsPreview
