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
// exports.createSchemaCustomization = ({ actions }) => {
// 	const { createTypes } = actions
// 	const typeDefs = `
//         type MarkdownRemarkFrontmatter {
//             title: String
//             category: String
//             slug: String
//             date: Date @dateformat
//             featuredImage: File @fileByRelativePath
//         }
//     `
// 	createTypes(typeDefs)
// }

const { createFilePath } = require("gatsby-source-filesystem")
exports.onCreateNode = ({ node, actions, getNode }) => {
	const { createNodeField } = actions
	//   fmImagesToRelative(node)
	if (node.internal.type === `MarkdownRemark`) {
		const value = createFilePath({ node, getNode })
		createNodeField({
			name: `slug`,
			node,
			value,
		})
	}
}
