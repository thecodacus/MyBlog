import React from "react"
import PropTypes from "prop-types"
import Article from "../../components/Article"

const PostDetailsPreview = ({ entry, widgetFor }) => {
	//  id, html, excerpt, title, category, featuredImage, date, publicURL
	const title = entry.getIn(["data", "title"])
	const date = entry.getIn(["data", "date"])
	const category = entry.getIn(["data", "category"])
	const featuredImage = entry.getIn(["data", "featuredImage"])
	const publicURL = featuredImage
	const html = widgetFor("body")
	return (
		<Article
			{...{
				// id: entry.getIn(["data", "id"]),
				html,
				// excerpt: entry.getIn(["data", "excerpt"]),
				title,
				category,
				featuredImage,
				date,
				publicURL,
			}}
		/>
	)
}

PostDetailsPreview.propTypes = {
	entry: PropTypes.shape({
		getIn: PropTypes.func,
	}),
	widgetFor: PropTypes.func,
}
export default PostDetailsPreview
