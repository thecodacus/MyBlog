const path = require("path")

exports.createPages = async ({ graphql, actions }) => {
	const { data } = await graphql(`
		query posts {
			allMarkdownRemark {
				nodes {
					fields {
						slug
					}
				}
			}
		}
	`)
	const nodes = data.allMarkdownRemark.nodes
	nodes.forEach(node => {
		actions.createPage({
			path: `${node.fields?.slug}`,
			component: path.resolve("./src/templates/post-details.js"),
			context: { slug: node.fields?.slug },
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
		let value = createFilePath({ node, getNode })
		value = value.split(" ").join("-")
		if (value[0] == "/" || value[0] == "\\") value = value.substr(1)
		createNodeField({
			name: `slug`,
			node,
			value: "/posts/" + value,
		})
	}
}
