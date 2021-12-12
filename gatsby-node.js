const path = require("path")
exports.createPages = async ({ graphql, actions }) => {
	const { data } = await graphql(`
		query posts {
			allMarkdownRemark {
				nodes {
					frontmatter {
						slug
					}
				}
			}
		}
	`)
	const nodes = data.allMarkdownRemark.nodes
	nodes.forEach(node => {
		actions.createPage({
			path: `/posts/${node.frontmatter.slug}`,
			component: path.resolve("./src/templates/post-details.js"),
			context: { slug: node.frontmatter.slug },
		})
	})
}

const { fmImagesToRelative } = require("gatsby-remark-relative-images")

exports.onCreateNode = ({ node }) => {
	fmImagesToRelative(node)
}
