import React from "react"
import PropTypes from "prop-types"
import { PostDetails } from "../../templates/post-details"

const PostDetailsPreview = ({ entry, widgetFor }) => {
	const tags = entry.getIn(["data", "tags"])
	return (
		<PostDetails
			content={widgetFor("body")}
			description={entry.getIn(["data", "description"])}
			tags={tags && tags.toJS()}
			title={entry.getIn(["data", "title"])}
		/>
	)
}

PostDetailsPreview.propTypes = {
	entry: PropTypes.shape({
		getIn: PropTypes.func,
	}),
	widgetFor: PropTypes.func,
}

export default BlogPostPreview
